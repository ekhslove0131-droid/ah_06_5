#!/usr/bin/env python3
"""Fit-free composition of two independently frozen Stack7 peer winners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.metrics import f1_score


LOG_FLOOR = 1e-12
EVIDENCE = "TWO_FROZEN_PEERS_CACHED_BASE_META_CV_ADAPTIVE_NOT_NESTED"
INNER_KEYS = ("ids", "y", "groups", "inner_folds", "outer_folds", "class_order", "probability")
FULL_KEYS = ("ids", "y", "groups", "folds", "class_order", "test_ids", "oof_probability", "test_probability")
PACKAGE_ROOT = Path(__file__).resolve().parent


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def probability_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def sequence_sha256(value):
    return canonical_sha256(np.asarray(value).tolist())


def _valid_sha(value):
    try:
        return isinstance(value, str) and len(value) == 64 and int(value, 16) >= 0
    except ValueError:
        return False


def _read_json(path, label="JSON artifact"):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected object")
    return value


def _write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact changed: {path.name}")
        return
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists():
        if not partial.is_file() or partial.read_bytes() != payload:
            raise ValueError(f"foreign unbound partial retained: {partial.name}")
    else:
        with partial.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(partial, path)


def _validate_probability(value, rows=None, classes=None, label="probability"):
    try:
        probability = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if probability.ndim != 2 or probability.shape[0] < 1 or probability.shape[1] < 2:
        raise ValueError(f"{label} dimensions are invalid")
    if rows is not None and probability.shape[0] != rows:
        raise ValueError(f"{label} row count changed")
    if classes is not None and probability.shape[1] != classes:
        raise ValueError(f"{label} class count changed")
    if (not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(1), 1.0, rtol=0.0, atol=1e-8)):
        raise ValueError(f"{label} values are invalid")
    return probability


def blend_probability(linear, nonlinear, alpha, pooling):
    """Blend frozen linear/nonlinear probabilities with exact endpoints."""
    left = _validate_probability(linear, label="linear probability")
    right = _validate_probability(nonlinear, *left.shape, label="nonlinear probability")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError("alpha must be a finite real")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if pooling not in {"arithmetic", "geometric"}:
        raise ValueError("pooling must be arithmetic or geometric")
    if alpha == 0.0:
        return left.copy()
    if alpha == 1.0:
        return right.copy()
    if pooling == "arithmetic":
        output = (1.0 - alpha) * left + alpha * right
    else:
        output = np.exp(
            (1.0 - alpha) * np.log(np.clip(left, LOG_FLOOR, 1.0))
            + alpha * np.log(np.clip(right, LOG_FLOOR, 1.0))
        )
    output /= output.sum(axis=1, keepdims=True)
    return _validate_probability(output, *left.shape, label="blended probability")


def _metrics(y, probability):
    y = np.asarray(y)
    classes = probability.shape[1]
    if y.ndim != 1 or y.shape[0] != probability.shape[0] or y.dtype.kind not in "iu":
        raise ValueError("labels are invalid")
    if set(np.unique(y).tolist()) != set(range(classes)):
        raise ValueError("labels must support every class")
    per_class = f1_score(y, probability.argmax(1), labels=np.arange(classes), average=None, zero_division=0)
    return float(np.mean(per_class)), [float(value) for value in per_class]


def search_probabilities(y, linear, nonlinear):
    linear = _validate_probability(linear, label="linear probability")
    nonlinear = _validate_probability(nonlinear, *linear.shape, label="nonlinear probability")
    rows = []
    for k in range(101):
        alpha = k / 100.0
        canonical_index = k * 2
        for pooling_index, pooling in enumerate(("arithmetic", "geometric")):
            declared_index = canonical_index + pooling_index
            endpoint_alias = pooling == "geometric" and k in (0, 100)
            probability = blend_probability(linear, nonlinear, alpha, pooling)
            macro, per_class = _metrics(y, probability)
            rows.append({
                "declared_index": declared_index,
                "alpha": alpha,
                "pooling": pooling,
                "alias_of": canonical_index if endpoint_alias else None,
                "endpoint": "linear" if k == 0 else "nonlinear" if k == 100 else None,
                "macro_f1": macro,
                "per_class_f1": per_class,
                "probability_sha256": probability_sha256(probability),
            })
    winner = rows[0]
    for row in rows[1:]:
        if row["macro_f1"] > winner["macro_f1"]:
            winner = row
    return rows, dict(winner)


def load_inner_export(npz_path, receipt_path, role, expected=None):
    npz_path, receipt_path = Path(npz_path), Path(receipt_path)
    receipt = _read_json(receipt_path, f"{role} inner export receipt")
    required = {
        "schema_version", "status", "role", "own_source_sha256", "own_runtime_config_sha256",
        "parent_source_sha256", "parent_runtime_config_sha256", "parent_selection_sha256",
        "parent_search_complete_sha256", "selected_inner_probability_sha256", "npz_sha256",
        "test_read", "full_bank_read", "fit_count",
    }
    if (set(receipt) != required or receipt["schema_version"] != "STACK7_CROSSBLEND_INNER_EXPORT_V1"
            or receipt["status"] != "COMPLETE" or receipt["role"] != role
            or receipt["npz_sha256"] != file_sha256(npz_path)
            or receipt["test_read"] is not False or receipt["full_bank_read"] is not False
            or receipt["fit_count"] != 0):
        raise ValueError(f"{role} inner export receipt is invalid")
    for key in (
        "own_source_sha256", "own_runtime_config_sha256", "parent_source_sha256",
        "parent_runtime_config_sha256", "parent_selection_sha256",
        "parent_search_complete_sha256", "selected_inner_probability_sha256", "npz_sha256",
    ):
        if not _valid_sha(receipt[key]):
            raise ValueError(f"{role} parent/export hash is invalid: {key}")
    if expected and any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{role} parent/export binding mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as stored:
            if set(stored.files) != set(INNER_KEYS) or "probability" not in stored.files:
                raise ValueError("inner export must contain exact probability member")
            arrays = {key: np.asarray(stored[key]) for key in INNER_KEYS}
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"{role} inner probability artifact is invalid") from exc
    rows, classes = len(arrays["ids"]), len(arrays["class_order"])
    probability = _validate_probability(arrays["probability"], rows, classes)
    for key in ("y", "groups", "inner_folds", "outer_folds"):
        if arrays[key].shape != (rows,):
            raise ValueError(f"{role} inner identity dimensions are invalid")
    if arrays["y"].dtype.kind not in "iu" or len(set(arrays["ids"].astype(str).tolist())) != rows:
        raise ValueError(f"{role} inner identity values are invalid")
    if probability_sha256(probability) != receipt["selected_inner_probability_sha256"]:
        raise ValueError(f"{role} selected inner probability hash mismatch")
    arrays["probability"] = probability
    arrays["receipt"] = receipt
    arrays["npz_sha256"] = receipt["npz_sha256"]
    arrays["receipt_sha256"] = file_sha256(receipt_path)
    return arrays


def validate_same_inner_identity(linear, nonlinear):
    for key in ("ids", "y", "groups", "inner_folds", "outer_folds", "class_order"):
        if not np.array_equal(linear[key], nonlinear[key]):
            raise ValueError(f"peer inner identity mismatch: {key}")
    return True


def _search_bindings(config, linear, nonlinear):
    return {
        "own_source_sha256": config["own_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "linear_export_npz_sha256": linear["npz_sha256"],
        "linear_export_receipt_sha256": linear["receipt_sha256"],
        "nonlinear_export_npz_sha256": nonlinear["npz_sha256"],
        "nonlinear_export_receipt_sha256": nonlinear["receipt_sha256"],
        "ids_sha256": sequence_sha256(linear["ids"]),
        "y_sha256": sequence_sha256(linear["y"]),
        "groups_sha256": sequence_sha256(linear["groups"]),
        "inner_folds_sha256": sequence_sha256(linear["inner_folds"]),
        "outer_folds_sha256": sequence_sha256(linear["outer_folds"]),
        "class_order_sha256": sequence_sha256(linear["class_order"]),
    }


def _verified_cached_inner_export(config, role, worker):
    """Re-run the frozen parent verifier and match its current INNER winner to cache."""
    spec = _parent_spec(config, role)
    expected_common = {
        "own_source_sha256": config["own_source_sha256"],
        "own_runtime_config_sha256": config["_runtime_config_sha256"],
        "parent_source_sha256": spec["expected_source_sha256"],
        "parent_runtime_config_sha256": spec["expected_runtime_config_sha256"],
    }
    cached = load_inner_export(
        config[f"{role}_inner_npz"], config[f"{role}_inner_receipt_json"], role,
        expected_common,
    )
    with tempfile.TemporaryDirectory(prefix=f"crossblend-{role}-reverify-") as temporary:
        current_npz, current_receipt = worker(config, role, "inner", temporary)
        current = load_inner_export(current_npz, current_receipt, role, {
            "parent_source_sha256": spec["expected_source_sha256"],
            "parent_runtime_config_sha256": spec["expected_runtime_config_sha256"],
        })
        validate_same_inner_identity(cached, current)
        receipt_keys = (
            "parent_selection_sha256", "parent_search_complete_sha256",
            "selected_inner_probability_sha256",
        )
        if (not np.array_equal(cached["probability"], current["probability"])
                or any(cached["receipt"].get(key) != current["receipt"].get(key) for key in receipt_keys)):
            raise ValueError(f"{role} cached export differs from current parent frozen state")
    return cached


def _expected_search_artifacts(config, linear, nonlinear):
    validate_same_inner_identity(linear, nonlinear)
    bindings = _search_bindings(config, linear, nonlinear)
    rows, winner = search_probabilities(linear["y"], linear["probability"], nonlinear["probability"])
    rows = [{**row, "bindings": bindings, "validation_evidence": EVIDENCE} for row in rows]
    winner = rows[winner["declared_index"]]
    ledger_payload = b"".join(_json_bytes(row) for row in rows)
    selection = {
        "schema_version": "STACK7_CROSSBLEND_FROZEN_SELECTION_V1", "status": "FROZEN",
        "declared_index": winner["declared_index"], "alpha": winner["alpha"],
        "pooling": winner["pooling"], "inner_macro_f1": winner["macro_f1"],
        "probability_sha256": winner["probability_sha256"], "bindings": bindings,
        "ledger_sha256": hashlib.sha256(ledger_payload).hexdigest(),
        "validation_evidence": EVIDENCE, "fit_count": 0,
        "full_read": False, "test_read": False,
    }
    complete = {
        "schema_version": "STACK7_CROSSBLEND_SEARCH_COMPLETE_V1", "status": "COMPLETE",
        "declared_rows": 202, "unique_effective_rows": 200, "endpoint_aliases": 2,
        "selection_sha256": hashlib.sha256(_json_bytes(selection)).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger_payload).hexdigest(),
        "selected_inner_macro_f1": winner["macro_f1"], "fit_count": 0,
        "full_read": False, "test_read": False, "validation_evidence": EVIDENCE,
    }
    return ledger_payload, selection, complete


def run_search(config, run_root, worker=None):
    """Search only two currently reverified frozen INNER peers; never full/test."""
    worker = _run_parent_worker if worker is None else worker
    linear = _verified_cached_inner_export(config, "linear", worker)
    nonlinear = _verified_cached_inner_export(config, "nonlinear", worker)
    ledger_payload, selection, complete = _expected_search_artifacts(config, linear, nonlinear)
    root = Path(run_root); root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "SCORE_LEDGER.jsonl"
    _write_once(ledger_path, ledger_payload)
    selection_path = root / "FROZEN_SELECTION.json"
    _write_once(selection_path, _json_bytes(selection))
    complete_path = root / "SEARCH_COMPLETE.json"
    _write_once(complete_path, _json_bytes(complete))
    return _read_json(complete_path, "crossblend search completion")


def submission_csv_bytes(test_ids, probability, class_order):
    test_ids = np.asarray(test_ids)
    class_order = np.asarray(class_order)
    probability = _validate_probability(probability, len(test_ids), len(class_order))
    if test_ids.ndim != 1 or class_order.ndim != 1 or len(set(test_ids.astype(str))) != len(test_ids):
        raise ValueError("submission identity is invalid")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["ID", "SUBCLASS"])
    labels = class_order.astype(str)[probability.argmax(1)]
    writer.writerows(zip(test_ids.astype(str).tolist(), labels.tolist()))
    return stream.getvalue().encode("utf-8")


def compose_deployment(linear, nonlinear, alpha, pooling):
    for key in ("ids", "y", "groups", "folds", "class_order", "test_ids"):
        if not np.array_equal(linear[key], nonlinear[key]):
            raise ValueError(f"peer deployment identity mismatch: {key}")
    rows, classes = len(linear["ids"]), len(linear["class_order"])
    tests = len(linear["test_ids"])
    linear_oof = _validate_probability(linear["oof_probability"], rows, classes, "linear OOF")
    nonlinear_oof = _validate_probability(nonlinear["oof_probability"], rows, classes, "nonlinear OOF")
    linear_test = _validate_probability(linear["test_probability"], tests, classes, "linear test")
    nonlinear_test = _validate_probability(nonlinear["test_probability"], tests, classes, "nonlinear test")
    oof = blend_probability(linear_oof, nonlinear_oof, alpha, pooling)
    test = blend_probability(linear_test, nonlinear_test, alpha, pooling)
    macro, per_class = _metrics(np.asarray(linear["y"]), oof)
    return {
        "ids": np.asarray(linear["ids"]), "y": np.asarray(linear["y"]),
        "groups": np.asarray(linear["groups"]), "folds": np.asarray(linear["folds"]),
        "class_order": np.asarray(linear["class_order"]), "test_ids": np.asarray(linear["test_ids"]),
        "oof_probability": oof, "test_probability": test,
        "adaptive_full_oof_macro_f1": macro, "adaptive_full_oof_per_class_f1": per_class,
    }


def _source_identity(root):
    root = Path(root)
    names = [line for line in (root / "SOURCE_ALLOWLIST.txt").read_text().splitlines() if line]
    if len(names) != len(set(names)):
        raise ValueError("source allowlist contains duplicates")
    rows = []
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise ValueError("source allowlist entry is invalid")
        rows.append({"path": name, "sha256": file_sha256(root / path)})
    return canonical_sha256(rows), rows


def load_runtime_config(path):
    config = _read_json(path, "runtime config")
    config["_runtime_config_sha256"] = file_sha256(path)
    config["_runtime_config_path"] = str(Path(path).resolve())
    return config


def verify_package(config, config_path, package_root=PACKAGE_ROOT):
    source_sha, rows = _source_identity(package_root)
    if config.get("own_source_sha256") != source_sha:
        raise ValueError("crossblend source identity changed")
    manifest_path = Path(config.get("source_manifest_json", ""))
    manifest = _read_json(manifest_path, "source manifest")
    if (file_sha256(manifest_path) != config.get("source_manifest_sha256")
            or manifest.get("source_sha256") != source_sha or manifest.get("files") != rows
            or file_sha256(config_path) != config["_runtime_config_sha256"]):
        raise ValueError("crossblend source/config manifest binding changed")
    return {"status": "PACKAGE_VERIFIED", "source_sha256": source_sha, "source_files": len(rows)}


def _parent_spec(config, role):
    spec = config.get("parents", {}).get(role)
    if not isinstance(spec, dict):
        raise ValueError(f"missing parent specification: {role}")
    required = {"source_root", "runtime_config_path", "expected_source_sha256", "expected_runtime_config_sha256"}
    if set(spec) != required or not _valid_sha(spec["expected_source_sha256"]) or not _valid_sha(spec["expected_runtime_config_sha256"]):
        raise ValueError(f"invalid parent specification: {role}")
    return spec


def _run_parent_worker(config, role, mode, output_root):
    spec = _parent_spec(config, role)
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), "_parent-worker",
        "--mode", mode, "--role", role, "--parent-root", spec["source_root"],
        "--parent-config", spec["runtime_config_path"],
        "--expected-source", spec["expected_source_sha256"],
        "--expected-config", spec["expected_runtime_config_sha256"],
        "--out", str(output_root),
    ]
    try:
        subprocess.run(command, check=True, cwd=spec["source_root"])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"verified parent worker failed: {role}/{mode}") from exc
    return Path(output_root) / "parent.npz", Path(output_root) / "parent.json"


def _publish_npz_and_receipt(arrays, receipt, npz_path, receipt_path):
    npz_path, receipt_path = Path(npz_path), Path(receipt_path)
    if receipt_path.is_file():
        loaded = _read_json(receipt_path)
        if loaded != receipt or loaded.get("npz_sha256") != file_sha256(npz_path):
            raise ValueError("immutable completed export changed")
        return loaded
    if npz_path.exists() or npz_path.with_name(f".{npz_path.name}.partial").exists():
        raise ValueError("unbound export artifact retained")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    partial = npz_path.with_name(f".{npz_path.name}.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    receipt = {**receipt, "npz_sha256": file_sha256(partial)}
    os.replace(partial, npz_path)
    _write_once(receipt_path, _json_bytes(receipt))
    return receipt


def export_inner(config, role, worker=_run_parent_worker):
    spec = _parent_spec(config, role)
    npz_path = Path(config[f"{role}_inner_npz"])
    receipt_path = Path(config[f"{role}_inner_receipt_json"])
    expected = {
        "own_source_sha256": config["own_source_sha256"],
        "own_runtime_config_sha256": config["_runtime_config_sha256"],
        "parent_source_sha256": spec["expected_source_sha256"],
        "parent_runtime_config_sha256": spec["expected_runtime_config_sha256"],
    }
    if receipt_path.is_file():
        return _verified_cached_inner_export(config, role, worker)["receipt"]
    with tempfile.TemporaryDirectory(prefix=f"crossblend-{role}-inner-") as temporary:
        worker_npz, worker_receipt = worker(config, role, "inner", temporary)
        raw = load_inner_export(worker_npz, worker_receipt, role, {
            "parent_source_sha256": spec["expected_source_sha256"],
            "parent_runtime_config_sha256": spec["expected_runtime_config_sha256"],
        })
        arrays = {key: raw[key] for key in INNER_KEYS}
        parent = raw["receipt"]
        receipt = {
            "schema_version": "STACK7_CROSSBLEND_INNER_EXPORT_V1", "status": "COMPLETE",
            "role": role, **expected,
            "parent_selection_sha256": parent["parent_selection_sha256"],
            "parent_search_complete_sha256": parent["parent_search_complete_sha256"],
            "selected_inner_probability_sha256": probability_sha256(arrays["probability"]),
            "test_read": False, "full_bank_read": False, "fit_count": 0,
        }
        return _publish_npz_and_receipt(arrays, receipt, npz_path, receipt_path)


def _load_parent_full(npz_path, receipt_path, role, spec):
    receipt = _read_json(receipt_path, f"{role} full export receipt")
    if (receipt.get("schema_version") != "STACK7_CROSSBLEND_PARENT_FULL_V1"
            or receipt.get("status") != "COMPLETE" or receipt.get("role") != role
            or receipt.get("parent_source_sha256") != spec["expected_source_sha256"]
            or receipt.get("parent_runtime_config_sha256") != spec["expected_runtime_config_sha256"]
            or receipt.get("npz_sha256") != file_sha256(npz_path) or receipt.get("fit_count") != 0):
        raise ValueError(f"{role} parent full export binding mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as stored:
            if set(stored.files) != set(FULL_KEYS):
                raise ValueError("parent full members changed")
            arrays = {key: np.asarray(stored[key]) for key in FULL_KEYS}
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"{role} parent full artifact invalid") from exc
    arrays["receipt"] = receipt
    arrays["npz_sha256"] = receipt["npz_sha256"]
    arrays["receipt_sha256"] = file_sha256(receipt_path)
    return arrays


def _verify_cross_selection(config, worker=None):
    worker = _run_parent_worker if worker is None else worker
    root = Path(config["cross_search_run_root"])
    selection_path, complete_path, ledger_path = root / "FROZEN_SELECTION.json", root / "SEARCH_COMPLETE.json", root / "SCORE_LEDGER.jsonl"
    selection = _read_json(selection_path, "crossblend frozen selection")
    complete = _read_json(complete_path, "crossblend search completion")
    linear = _verified_cached_inner_export(config, "linear", worker)
    nonlinear = _verified_cached_inner_export(config, "nonlinear", worker)
    expected_ledger, expected_selection, expected_complete = _expected_search_artifacts(
        config, linear, nonlinear,
    )
    if (ledger_path.read_bytes() != expected_ledger
            or selection != expected_selection or complete != expected_complete
            or complete.get("selection_sha256") != file_sha256(selection_path)
            or complete.get("ledger_sha256") != file_sha256(ledger_path)):
        raise ValueError("crossblend ledger/selection/winner correspondence mismatch")
    return selection, {"selection_sha256": file_sha256(selection_path), "search_complete_sha256": file_sha256(complete_path), "ledger_sha256": file_sha256(ledger_path)}


def _load_canonical_reference(config):
    path = Path(config["canonical_full_npz"])
    if file_sha256(path) != config["canonical_full_npz_sha256"]:
        raise ValueError("canonical full reference hash mismatch")
    try:
        with np.load(path, allow_pickle=False) as stored:
            values = {
                "ids": np.asarray(stored["full_ids"]), "y": np.asarray(stored["full_y"]),
                "groups": np.asarray(stored["full_groups"]), "folds": np.asarray(stored["full_outer_folds"]),
                "class_order": np.asarray(stored["class_order"]),
            }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("canonical full reference members invalid") from exc
    sample = Path(config["sample_submission_csv"])
    if file_sha256(sample) != config["sample_submission_sha256"]:
        raise ValueError("canonical sample submission hash mismatch")
    with sample.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ID", "SUBCLASS"]:
            raise ValueError("canonical sample submission columns changed")
        values["test_ids"] = np.asarray([row["ID"] for row in reader])
    return values


def _publish_deployment(root, arrays, receipt):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    npz_path, csv_path, pending_path, complete_path = root / "predictions.npz", root / "submission_stack7_crossblend_frozen.csv", root / "OUTPUT_PENDING.json", root / "COMPLETE.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        base = dict(complete); expected_npz = base.pop("predictions_sha256", None); expected_csv = base.pop("submission_sha256", None)
        if base != receipt or file_sha256(npz_path) != expected_npz or file_sha256(csv_path) != expected_csv:
            raise ValueError("immutable completed deployment changed")
        return complete
    npz_partial, csv_partial = root / ".predictions.npz.partial", root / ".submission.csv.partial"
    if any(path.exists() for path in (npz_path, csv_path, pending_path, npz_partial, csv_partial)):
        raise ValueError("unbound deployment artifacts retained")
    with npz_partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    csv_payload = submission_csv_bytes(arrays["test_ids"], arrays["test_probability"], arrays["class_order"])
    with csv_partial.open("wb") as stream:
        stream.write(csv_payload); stream.flush(); os.fsync(stream.fileno())
    pending = {"schema_version": "STACK7_CROSSBLEND_OUTPUT_PENDING_V1", "receipt_sha256": canonical_sha256(receipt), "predictions_sha256": file_sha256(npz_partial), "submission_sha256": file_sha256(csv_partial)}
    _write_once(pending_path, _json_bytes(pending))
    os.replace(npz_partial, npz_path); os.replace(csv_partial, csv_path)
    complete = {**receipt, "predictions_sha256": pending["predictions_sha256"], "submission_sha256": pending["submission_sha256"]}
    _write_once(complete_path, _json_bytes(complete))
    return complete


def _reuse_completed_deployment(config, run_root, selection, search_binding):
    root = Path(run_root); complete_path = root / "COMPLETE.json"
    if not complete_path.is_file():
        return None
    complete = _read_json(complete_path, "crossblend deployment completion")
    npz_path, csv_path = root / "predictions.npz", root / "submission_stack7_crossblend_frozen.csv"
    if (complete.get("schema_version") != "STACK7_CROSSBLEND_DEPLOYMENT_COMPLETE_V1"
            or complete.get("status") != "COMPLETE" or complete.get("fit_count") != 0
            or complete.get("selected_alpha") != selection["alpha"]
            or complete.get("selected_pooling") != selection["pooling"]
            or complete.get("own_source_sha256") != config["own_source_sha256"]
            or complete.get("runtime_config_sha256") != config["_runtime_config_sha256"]
            or complete.get("search_binding") != search_binding
            or complete.get("canonical_full_npz_sha256") != config["canonical_full_npz_sha256"]
            or complete.get("sample_submission_sha256") != config["sample_submission_sha256"]
            or complete.get("predictions_sha256") != file_sha256(npz_path)
            or complete.get("submission_sha256") != file_sha256(csv_path)):
        raise ValueError("completed crossblend deployment binding mismatch")
    return complete


def run_deploy(config, run_root, worker=_run_parent_worker):
    selection, search_binding = _verify_cross_selection(config, worker)  # before full/test
    reused = _reuse_completed_deployment(config, run_root, selection, search_binding)
    if reused is not None:
        return reused
    parents = {}
    with tempfile.TemporaryDirectory(prefix="crossblend-parent-full-") as temporary:
        for role in ("linear", "nonlinear"):
            target = Path(temporary) / role; target.mkdir()
            npz_path, receipt_path = worker(config, role, "deployment", target)
            parents[role] = _load_parent_full(npz_path, receipt_path, role, _parent_spec(config, role))
        validate_same = compose_deployment(parents["linear"], parents["nonlinear"], selection["alpha"], selection["pooling"])
        canonical = _load_canonical_reference(config)
        for role, values in parents.items():
            for key in ("ids", "y", "groups", "folds", "class_order", "test_ids"):
                if not np.array_equal(values[key], canonical[key]):
                    raise ValueError(f"{role} deployment/canonical identity mismatch: {key}")
        arrays = {key: validate_same[key] for key in ("ids", "y", "groups", "folds", "class_order", "test_ids", "oof_probability", "test_probability")}
        receipt = {
            "schema_version": "STACK7_CROSSBLEND_DEPLOYMENT_COMPLETE_V1", "status": "COMPLETE",
            "selected_alpha": selection["alpha"], "selected_pooling": selection["pooling"],
            "selected_inner_macro_f1": selection["inner_macro_f1"],
            "adaptive_full_oof_macro_f1": validate_same["adaptive_full_oof_macro_f1"],
            "adaptive_full_oof_per_class_f1": validate_same["adaptive_full_oof_per_class_f1"],
            "recipe": "blend_after_parent_averaging", "fit_count": 0,
            "full_score_used_for_selection": False, "test_used_for_selection": False,
            "website_submitted": False, "validation_evidence": EVIDENCE,
            "own_source_sha256": config["own_source_sha256"], "runtime_config_sha256": config["_runtime_config_sha256"],
            "search_binding": search_binding,
            "parent_bindings": {
                role: {"npz_sha256": values["npz_sha256"], "receipt_sha256": values["receipt_sha256"], **{key: values["receipt"].get(key) for key in ("parent_source_sha256", "parent_runtime_config_sha256", "parent_selection_sha256", "parent_search_complete_sha256", "parent_deployment_complete_sha256")}}
                for role, values in parents.items()
            },
            "canonical_full_npz_sha256": config["canonical_full_npz_sha256"],
            "sample_submission_sha256": config["sample_submission_sha256"],
        }
        return _publish_deployment(run_root, arrays, receipt)


def _parent_worker(args):
    """Isolated import namespace for one immutable parent package."""
    parent_root, config_path, output = Path(args.parent_root), Path(args.parent_config), Path(args.out)
    if file_sha256(config_path) != args.expected_config:
        raise ValueError("parent runtime config hash mismatch")
    sys.path.insert(0, str(parent_root))
    try:
        import cli as parent_cli
        import meta_deployment as parent_deployment
        import meta_search as parent_search
        from artifact_store import load_full_bank, load_or_fit_head
        from meta_core import blend
        config = parent_cli.load_runtime_config(config_path)
        if config.get("meta_source_sha256") != args.expected_source:
            raise ValueError("parent source hash mismatch")
        parent_cli.verify_package(config, config_path, parent_root)
        resolved = parent_cli.resolve_deployment_config(config)
        selection, binding = parent_deployment.verify_frozen_selection(resolved)
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise ValueError("parent worker output directory is not empty")
        else:
            output.mkdir(parents=True)
        if args.mode == "inner":
            bank, baseline, guard, _lineage, _bindings = parent_deployment._verified_inner_context(resolved)
            if not selection["deployment_required"]:
                probability = baseline
            else:
                folds = bank["inner_folds"].astype(np.int64)
                probability_meta = np.zeros_like(baseline); coverage = np.zeros(len(folds), dtype=np.uint8)
                receipts = selection["selected_fold_receipts"]
                for fold in (0, 1, 2):
                    valid = folds == fold
                    identity = parent_search._fit_identity(resolved, bank, selection["selected_declaration"], fold)
                    def forbidden_fit(): raise ValueError("crossblend parent export attempted fit")
                    def forbidden_predict(_): raise ValueError("crossblend parent export attempted prediction")
                    _model, valid_probability, receipt = load_or_fit_head(Path(resolved["meta_search_run_root"]) / "fits", identity, forbidden_fit, forbidden_predict)
                    selected = next(item for item in receipts if item["fold"] == fold)
                    if receipt["valid_probability_sha256"] != selected["valid_probability_sha256"]:
                        raise ValueError("parent selected fold probability binding mismatch")
                    probability_meta[valid] = valid_probability; coverage[valid] += 1
                if not np.all(coverage == 1):
                    raise ValueError("parent selected inner coverage mismatch")
                probability = blend(baseline, probability_meta, selection["selected_alpha"], selection["selected_pooling"], guard)
            if probability_sha256(probability) != selection["selected_probability_sha256"] or probability_sha256(probability) != binding["selected_inner_probability_sha256"]:
                raise ValueError("parent selected inner probability does not reproduce")
            arrays = {"ids": bank["inner_ids"], "y": bank["inner_y"], "groups": bank["inner_groups"], "inner_folds": bank["inner_folds"], "outer_folds": bank["inner_outer_folds"], "class_order": bank["class_order"], "probability": probability}
            npz_path = output / "parent.npz"; np.savez_compressed(npz_path, **arrays)
            receipt = {"schema_version": "STACK7_CROSSBLEND_INNER_EXPORT_V1", "status": "COMPLETE", "role": args.role, "own_source_sha256": "0" * 64, "own_runtime_config_sha256": "0" * 64, "parent_source_sha256": args.expected_source, "parent_runtime_config_sha256": args.expected_config, "parent_selection_sha256": binding["selection_sha256"], "parent_search_complete_sha256": binding["search_complete_sha256"], "selected_inner_probability_sha256": probability_sha256(probability), "npz_sha256": file_sha256(npz_path), "test_read": False, "full_bank_read": False, "fit_count": 0}
        else:
            bank = load_full_bank(resolved)
            test_ids = parent_deployment._load_sample(resolved)
            if selection["deployment_required"]:
                complete_path = Path(resolved["meta_deployment_run_root"]) / "COMPLETE.json"; predictions_path = Path(resolved["meta_deployment_run_root"]) / "predictions.npz"
                complete = _read_json(complete_path, "parent deployment completion")
                if (complete.get("status") != "COMPLETE" or complete.get("source_hashes", {}).get("meta_source_sha256") != args.expected_source or complete.get("source_hashes", {}).get("runtime_config_sha256") != args.expected_config or complete.get("cache_hashes", {}).get("selection_sha256") != binding["selection_sha256"] or complete.get("predictions_sha256") != file_sha256(predictions_path)):
                    raise ValueError("parent deployment completion binding mismatch")
                with np.load(predictions_path, allow_pickle=False) as stored:
                    arrays = {key: np.asarray(stored[key]) for key in FULL_KEYS}
                deployment_sha = file_sha256(complete_path)
            else:
                no_deploy_path = Path(resolved["meta_deployment_run_root"]) / "NO_DEPLOYMENT.json"
                no_deploy = _read_json(no_deploy_path, "parent no-deployment receipt")
                if no_deploy.get("status") != "COMPLETE" or no_deploy.get("selection_sha256") != binding["selection_sha256"] or no_deploy.get("test_read") is not False:
                    raise ValueError("parent no-deployment receipt mismatch")
                oof, test, _complete = parent_deployment._load_saved_v3(resolved, bank, test_ids)
                arrays = {"ids": bank["full_ids"], "y": bank["full_y"], "groups": bank["full_groups"], "folds": bank["full_outer_folds"], "class_order": bank["class_order"], "test_ids": test_ids, "oof_probability": oof, "test_probability": test}
                deployment_sha = file_sha256(no_deploy_path)
            npz_path = output / "parent.npz"; np.savez_compressed(npz_path, **arrays)
            receipt = {"schema_version": "STACK7_CROSSBLEND_PARENT_FULL_V1", "status": "COMPLETE", "role": args.role, "parent_source_sha256": args.expected_source, "parent_runtime_config_sha256": args.expected_config, "parent_selection_sha256": binding["selection_sha256"], "parent_search_complete_sha256": binding["search_complete_sha256"], "parent_deployment_complete_sha256": deployment_sha, "npz_sha256": file_sha256(npz_path), "fit_count": 0}
        (output / "parent.json").write_bytes(_json_bytes(receipt))
    finally:
        sys.path.pop(0)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="stack7-crossblend")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("export-inner", "search", "deploy"):
        current = sub.add_parser(command); current.add_argument("--config", required=True); current.add_argument("--start", action="store_true")
    worker = sub.add_parser("_parent-worker")
    for name in ("mode", "role", "parent-root", "parent-config", "expected-source", "expected-config", "out"):
        worker.add_argument(f"--{name}", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in {"export-inner", "search", "deploy"} and not args.start:
        print(json.dumps({"status": "START_FLAG_REQUIRED"}, sort_keys=True)); return 2
    try:
        if args.command == "_parent-worker":
            return _parent_worker(args)
        config_path = Path(args.config); config = load_runtime_config(config_path)
        verify_package(config, config_path)
        if args.command == "export-inner":
            result = {role: export_inner(config, role) for role in ("linear", "nonlinear")}
        elif args.command == "search":
            result = run_search(config, config["cross_search_run_root"])
        else:
            result = run_deploy(config, config["cross_deployment_run_root"])
        print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True)); return 3


if __name__ == "__main__":
    raise SystemExit(main())
