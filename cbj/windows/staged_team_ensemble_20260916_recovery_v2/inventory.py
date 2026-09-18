"""Deterministic expansion and fail-closed admission of 128 declarations."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re

from contracts import SPLIT_CONTRACT


HEX64 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_STATUS = {"EXECUTABLE", "BLOCKED", "INELIGIBLE"}
ALLOWED_DEVICE = {"GPU_REQUIRED", "CPU_INTENTIONAL"}


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _expand_family(source_kind: str, family: dict) -> list[dict]:
    family_id = family.get("id")
    grid = family.get("grid")
    if not isinstance(family_id, str) or not family_id or not isinstance(grid, dict) or not grid:
        raise ValueError("family identity or grid is invalid")
    keys = list(grid)
    if any(not isinstance(grid[key], list) or not grid[key] for key in keys):
        raise ValueError("grid axes must be nonempty lists")
    rows = []
    for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), start=1):
        parameters = dict(zip(keys, values))
        identity = {
            "schema_version": "TEAM_BASE_CANDIDATE_IDENTITY_V1",
            "source_kind": source_kind,
            "family": family_id,
            "parameters": parameters,
            "fixed": family.get("fixed", {}),
            "learner": family.get("learner"),
            "split_contract": SPLIT_CONTRACT,
        }
        rows.append({
            "candidate_id": f"{source_kind}-{family_id}-{index:04d}",
            "config_hash": _digest(identity),
            "source_kind": source_kind,
            "family": family_id,
            "parameters": parameters,
            "fixed": copy.deepcopy(family.get("fixed", {})),
            "learner": family.get("learner"),
        })
    return rows


def _support_record(config_hash: str, record: dict | None) -> tuple[str, list[str], dict]:
    if record is None:
        return "BLOCKED", ["ADAPTER_NOT_REGISTERED"], {}
    if not isinstance(record, dict) or record.get("status") not in ALLOWED_STATUS:
        raise ValueError("support status is invalid")
    status = record["status"]
    reasons = record.get("reasons", [])
    if not isinstance(reasons, list) or any(not isinstance(value, str) or not value for value in reasons):
        raise ValueError("support reasons are invalid")
    metadata = {key: copy.deepcopy(value) for key, value in record.items() if key not in {"status", "reasons"}}
    if status == "EXECUTABLE":
        if reasons:
            raise ValueError("executable support cannot contain reasons")
        if HEX64.fullmatch(metadata.get("implementation_source_hash", "")) is None:
            raise ValueError("executable support source hash is invalid")
        if metadata.get("device_kind") not in ALLOWED_DEVICE:
            raise ValueError("executable support device is invalid")
    elif not reasons:
        raise ValueError("blocked or ineligible support requires reasons")
    return status, reasons, metadata


def compile_admission(old_space: dict, team_space: dict, support: dict) -> dict:
    if not isinstance(support, dict):
        raise ValueError("support registry must be an object")
    raw = []
    for family in old_space.get("families", []):
        raw.extend(_expand_family("OLD", family))
    for family in team_space.get("families", []):
        raw.extend(_expand_family("TEAM", family))
    if len(raw) != 128:
        raise ValueError(f"declared inventory must contain 128 candidates, got {len(raw)}")
    config_hashes = {row["config_hash"] for row in raw}
    if len(config_hashes) != 128 or len({row["candidate_id"] for row in raw}) != 128:
        raise ValueError("candidate identities are duplicated")
    for config_hash, record in support.items():
        if not isinstance(record, dict) or record.get("status") not in ALLOWED_STATUS:
            raise ValueError("support status is invalid")
        if config_hash not in config_hashes:
            raise ValueError("support references unknown config")
    declarations = []
    counts = {"EXECUTABLE": 0, "BLOCKED": 0, "INELIGIBLE": 0}
    for row in raw:
        status, reasons, metadata = _support_record(row["config_hash"], support.get(row["config_hash"]))
        result = {**row, "status": status, "reasons": reasons, **metadata}
        declarations.append(result)
        counts[status] += 1
    report = {
        "schema_version": "TEAM_BASE_ADMISSION_REPORT_V1",
        "split_contract": SPLIT_CONTRACT,
        "declared_count": 128,
        "source_counts": {"OLD": 80, "TEAM": 48},
        "counts": counts,
        "declarations": declarations,
    }
    report["admission_sha256"] = _digest(report)
    return report

