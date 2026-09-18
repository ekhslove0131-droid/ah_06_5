"""Detached one-shot lifecycle and resumable phase state machine."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

from atomic_state import append_jsonl, atomic_write_json, read_json


PHASES = (
    "feature_state", "model", "inner_oof", "round_manifest", "outer_refit", "fold_completion",
)


class InjectedExit(RuntimeError):
    pass


RUNTIME_HASH_KEYS = {
    "source_sha256", "runtime_sha256", "data_sha256", "partition_sha256",
    "admission_sha256", "class_order_sha256",
}


def freeze_runtime(run_root, source_manifest):
    if set(source_manifest) != RUNTIME_HASH_KEYS or any(
        not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in source_manifest.values()
    ):
        raise ValueError("source manifest must contain the six immutable sha256 values")
    path = Path(run_root) / "IMMUTABLE_RUNTIME.json"
    payload = {"schema_version": "IMMUTABLE_RUNTIME_V1", **source_manifest}
    if path.is_file():
        if read_json(path) != payload:
            raise ValueError("source manifest changed during resume")
    else:
        atomic_write_json(path, payload)
    return payload


def _heartbeat(run_root, state):
    atomic_write_json(Path(run_root) / "HEARTBEAT.json", {
        "schema_version": "TEAM_RUN_HEARTBEAT_V1",
        "pid": os.getpid(),
        "status": state["status"],
        "phase": state.get("current_phase"),
        "candidate": state.get("candidate"),
        "outer_index": state.get("outer_index"),
        "inner_index": state.get("inner_index"),
        "subinner_index": state.get("subinner_index"),
        "completion_counts": state.get("completion_counts", {}),
        "latest_receipt": state.get("latest_receipt"),
    })


def run_resumable(run_root, callbacks, source_manifest, *, fail_after_phase=None):
    run_root = Path(run_root)
    if tuple(callbacks) != PHASES:
        raise ValueError("runner callbacks must match the frozen phase order")
    run_root.mkdir(parents=True, exist_ok=True)
    freeze_runtime(run_root, source_manifest)
    state_path = run_root / "STATE.json"
    events_path = run_root / "events.jsonl"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("source_manifest") != source_manifest:
            raise ValueError("source manifest changed during resume")
    else:
        state = {
            "schema_version": "TEAM_RUNNER_STATE_V1",
            "status": "RUNNING",
            "source_manifest": source_manifest,
            "completed_phases": [],
            "current_phase": None,
            "receipts": {},
            "candidate": None,
            "outer_index": None,
            "inner_index": None,
            "subinner_index": None,
            "completion_counts": {"phases": 0, "total_phases": len(PHASES)},
            "latest_receipt": None,
        }
        atomic_write_json(state_path, state)
        _heartbeat(run_root, state)
        append_jsonl(events_path, {"event": "RUN_INITIALIZED", "completed_phases": 0})
    completed = list(state["completed_phases"])
    for phase in PHASES:
        if phase in completed:
            continue
        state["current_phase"] = phase
        state["status"] = "RUNNING"
        atomic_write_json(state_path, state)
        _heartbeat(run_root, state)
        append_jsonl(events_path, {"event": "PHASE_START", "phase": phase})
        receipt = callbacks[phase]()
        if not isinstance(receipt, dict):
            raise ValueError("phase callback must return a receipt object")
        state["receipts"][phase] = receipt
        completed.append(phase)
        state["completed_phases"] = completed
        state["current_phase"] = None
        state["completion_counts"]["phases"] = len(completed)
        state["latest_receipt"] = receipt
        atomic_write_json(state_path, state)
        _heartbeat(run_root, state)
        append_jsonl(events_path, {"event": "PHASE_COMPLETE", "phase": phase})
        if fail_after_phase == phase:
            raise InjectedExit(f"injected exit after {phase}")
    state["status"] = "COMPLETE"
    state["current_phase"] = None
    atomic_write_json(state_path, state)
    _heartbeat(run_root, state)
    atomic_write_json(run_root / "COMPLETE.json", {
        "schema_version": "TEAM_RUN_COMPLETE_V1",
        "status": "COMPLETE",
        "completed_phases": list(PHASES),
        "source_manifest": source_manifest,
        "automatic_next_bundle": False,
        "submission": False,
    })
    append_jsonl(events_path, {"event": "RUN_COMPLETE", "completed_phases": len(PHASES)})
    return state


def _runner_live(run_root):
    lock_path = Path(run_root) / ".runner.lock"
    if not lock_path.is_file():
        return False
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def status(run_root):
    run_root = Path(run_root)
    if not run_root.exists():
        return {"status": "NOT_STARTED", "runner_live": False}
    state_path = run_root / "STATE.json"
    state = read_json(state_path) if state_path.is_file() else {"status": "STARTING"}
    return {**state, "runner_live": _runner_live(run_root)}


def launch_detached(run_root, worker, *, start=False):
    run_root = Path(run_root)
    if not start:
        return {"status": "START_FLAG_REQUIRED"}
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".runner.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return {"status": "ALREADY_RUNNING"}
        read_pipe, write_pipe = os.pipe()
        first_pid = os.fork()
        if first_pid == 0:
            try:
                os.close(read_pipe)
                os.setsid()
                second_pid = os.fork()
                if second_pid > 0:
                    os._exit(0)
                with os.fdopen(write_pipe, "w", closefd=True) as stream:
                    stream.write(str(os.getpid()))
                    stream.flush()
                log_descriptor = os.open(run_root / "runner.log", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                os.dup2(log_descriptor, 1)
                os.dup2(log_descriptor, 2)
                if log_descriptor > 2:
                    os.close(log_descriptor)
                try:
                    worker()
                except BaseException:
                    atomic_write_json(run_root / "FAILURE.json", {
                        "schema_version": "TEAM_RUN_FAILURE_V1",
                        "status": "FAILED",
                        "traceback": traceback.format_exc(),
                    })
                    raise
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(write_pipe)
        with os.fdopen(read_pipe, "r", closefd=True) as stream:
            grandchild_text = stream.read()
        os.waitpid(first_pid, 0)
        os.close(descriptor)
        if not grandchild_text.strip().isdigit():
            raise RuntimeError("detached runner failed before PID handoff")
        pid = int(grandchild_text)
        atomic_write_json(run_root / "LAUNCH.json", {
            "schema_version": "TEAM_RUN_LAUNCH_V1", "status": "STARTED", "pid": pid,
        })
        return {"status": "STARTED", "pid": pid}
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def configured_worker(config_path, run_root):
    config = read_json(config_path)
    config["_runtime_config_path"] = str(Path(config_path).resolve())
    config["_runtime_config_sha256"] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    entrypoint = config.get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise ValueError("runtime config entrypoint must be module:function")
    module_name, function_name = entrypoint.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    function(config, Path(run_root))
