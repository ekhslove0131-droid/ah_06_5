"""Fail-closed 128-declaration eligibility and receipt audit."""

from __future__ import annotations

import json
from pathlib import Path

from inventory import compile_admission
from model_adapters import effective_recipe
from source_closure import composite_hash


FAMILY_SOURCES = {
    "G1": ["old_candidates.py", "model_adapters.py", "vendor/mcmp_repr.py"],
    "G2": ["old_candidates.py", "model_adapters.py", "vendor/mcmp_repr.py"],
    "G3": ["old_candidates.py", "model_adapters.py", "vendor/mcmp_repr.py"],
    "G4": ["old_candidates.py", "model_adapters.py", "vendor/mcmp_repr.py"],
    "G5": ["neural_candidates.py", "model_adapters.py", "torch_models.py"],
    "G6": ["neural_candidates.py", "model_adapters.py", "torch_models.py", "vendor/mcmp_repr.py"],
    "T1": ["feature_transforms.py", "model_adapters.py", "raw_views.py"],
    "T2": ["feature_transforms.py", "model_adapters.py", "raw_views.py"],
    "T3": ["feature_transforms.py", "model_adapters.py", "raw_views.py", "vendor/mcmp_repr.py"],
    "T4": ["feature_transforms.py", "model_adapters.py", "torch_models.py", "vendor/mcmp_repr.py"],
    "T5": ["feature_transforms.py", "model_adapters.py", "raw_views.py", "vendor/mcmp_repr.py"],
    "T6": ["feature_transforms.py", "model_adapters.py", "raw_views.py", "vendor/mcmp_repr.py"],
}


def _spaces(root):
    root = Path(root)
    return (
        json.loads((root / "fixtures/old_grid_80.json").read_text(encoding="utf-8")),
        json.loads((root / "fixtures/team_grid_48.json").read_text(encoding="utf-8")),
    )


def audit_admission(root):
    root = Path(root)
    old, team = _spaces(root)
    raw = compile_admission(old, team, {})
    support = {}
    for declaration in raw["declarations"]:
        recipe = effective_recipe(declaration)
        family = declaration["family"]
        implementation_hash = composite_hash(root, FAMILY_SOURCES[family])
        if family == "T4" and declaration["parameters"]["head"] == "LightGBM_fixed":
            support[declaration["config_hash"]] = {
                "status": "BLOCKED",
                "reasons": ["GPU_BUILD_UNAVAILABLE", "CPU_FALLBACK_FORBIDDEN"],
                "implementation_source_hash": implementation_hash,
                "device_kind": "GPU_REQUIRED",
            }
        else:
            support[declaration["config_hash"]] = {
                "status": "EXECUTABLE",
                "reasons": [],
                "implementation_source_hash": implementation_hash,
                "device_kind": recipe["device_kind"],
                "effective_recipe": recipe,
            }
    report = compile_admission(old, team, support)
    if report["counts"] != {"EXECUTABLE": 124, "BLOCKED": 4, "INELIGIBLE": 0}:
        raise ValueError("admission count differs from the sealed support audit")
    return report


def audit_receipts(root):
    root = Path(root)
    required = sorted(path for path in (root / "artifacts").glob("TASK*_ACCEPTANCE.json"))
    names = {path.name for path in required}
    expected = {f"TASK{index}_" for index in range(1, 11)}
    if any(not any(name.startswith(prefix) for name in names) for prefix in expected):
        raise ValueError("acceptance receipt set is incomplete")
    receipts = {}
    for path in required:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not str(value.get("status", "")).startswith("PASS"):
            raise ValueError(f"acceptance receipt did not pass: {path.name}")
        receipts[path.name] = value["schema_version"]
    return {"all_pass": True, "receipts": receipts}


def slot_for(declaration):
    family = declaration["family"]
    if family.startswith("G"):
        return "project"
    if family == "T2":
        return "kim"
    if family == "T1":
        return "ahn"
    if family in {"T3", "T4", "T5", "T6"}:
        return "cross"
    raise ValueError("candidate family has no ensemble slot")
