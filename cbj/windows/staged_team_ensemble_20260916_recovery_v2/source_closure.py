"""Relocatable exact source inventory for the V4 bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_FILES = {
    "anchor_pipeline.py", "atomic_state.py", "base_bank.py", "base_receipts.py",
    "cache_store.py", "cli.py", "contracts.py", "eligibility_audit.py",
    "ensemble_grid.py", "feature_transforms.py", "inventory.py", "log_bias.py",
    "meta_rounds.py", "metrics.py", "model_adapters.py", "neural_candidates.py",
    "numeric_views.py", "old_candidates.py", "outer_procedure.py",
    "partition_contract.py", "promotion.py", "raw_views.py", "resource_gate.py",
    "runner.py", "slot_selector.py", "source_closure.py", "support_guard.py", "torch_models.py",
    "workflow.py",
}
ROOT_CONFIG_FILES = {"runtime_config.template.json", "runtime_config.windows.json", "README_KO.md"}
FIXTURE_FILES = {"fixtures/old_grid_80.json", "fixtures/team_grid_48.json"}
VENDOR_FILES = {"vendor/__init__.py", "vendor/mcmp_repr.py"}
TEST_FILES = {
    "tests/__init__.py", "tests/test_anchors.py", "tests/test_base_bank.py",
    "tests/test_cache_store.py", "tests/test_cli.py", "tests/test_end_to_end.py",
    "tests/test_ensemble_grid.py", "tests/test_feature_transforms.py",
    "tests/test_gpu_native.py", "tests/test_inventory.py", "tests/test_log_bias.py",
    "tests/test_meta_rounds.py", "tests/test_metrics.py", "tests/test_models.py",
    "tests/test_neural_candidates.py", "tests/test_numeric_views.py",
    "tests/test_old_candidates.py", "tests/test_outer_procedure.py",
    "tests/test_partitions.py", "tests/test_promotion.py", "tests/test_raw_views.py",
    "tests/test_resource_gate.py", "tests/test_runner.py", "tests/test_source_closure.py",
    "tests/test_torch_models.py",
}
ACCEPTANCE_FILES = {
    "artifacts/TASK1_INVENTORY_ACCEPTANCE.json",
    "artifacts/TASK2_PARTITION_ACCEPTANCE.json",
    "artifacts/TASK3_FEATURE_ACCEPTANCE.json",
    "artifacts/TASK3B_OLD_FAMILY_ACCEPTANCE.json",
    "artifacts/TASK4_MODEL_ACCEPTANCE.json",
    "artifacts/TASK5_ANCHOR_ACCEPTANCE.json",
    "artifacts/TASK6_BASE_BANK_ACCEPTANCE.json",
    "artifacts/TASK7_ROUND0_ACCEPTANCE.json",
    "artifacts/TASK8_PROMOTION_ACCEPTANCE.json",
    "artifacts/TASK9_OUTER_ACCEPTANCE.json",
    "artifacts/TASK10_RUNNER_ACCEPTANCE.json",
    "artifacts/TASK11_SOURCE_ACCEPTANCE.json",
    "artifacts/LIGHTGBM_GPU_BLOCKED.json",
}
GENERATED_CLOSURE_FILES = {
    "SOURCE_ALLOWLIST.txt", "SHA256SUMS.txt", "LOCAL_ACCEPTANCE.json",
    "artifacts/SOURCE_CLOSURE.json", "artifacts/TRANSFER_PLAN.json",
    "artifacts/WINDOWS_NATIVE_ACCEPTANCE.json", "artifacts/LAUNCH_RECEIPT.json",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composite_hash(root, relative_paths):
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(Path(root, relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _expected_existing(root):
    candidates = SOURCE_FILES | ROOT_CONFIG_FILES | FIXTURE_FILES | VENDOR_FILES | TEST_FILES | ACCEPTANCE_FILES
    return {relative for relative in candidates if (Path(root) / relative).is_file()}


def declared_payload_files(root):
    root = Path(root)
    expected = _expected_existing(root)
    scanned = set()
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in GENERATED_CLOSURE_FILES or relative.startswith(".git/"):
            continue
        if path.suffix in {".py", ".json", ".txt", ".md"}:
            scanned.add(relative)
    undeclared = sorted(scanned - expected)
    if undeclared:
        raise ValueError("undeclared source/config files: " + ",".join(undeclared))
    missing_required = sorted((SOURCE_FILES | FIXTURE_FILES | VENDOR_FILES | TEST_FILES | ACCEPTANCE_FILES) - expected)
    # workflow/end-to-end/README/windows config are introduced in later closure steps.
    deferred = {"workflow.py", "tests/test_end_to_end.py"}
    missing_required = [value for value in missing_required if value not in deferred]
    if missing_required:
        raise ValueError("required payload files missing: " + ",".join(missing_required))
    return sorted(expected)


def source_payload_files(root):
    """Return closure inputs excluding the self-referential Windows config."""
    return [relative for relative in declared_payload_files(root) if relative != "runtime_config.windows.json"]


def build_source_closure(root, allowlist):
    root = Path(root)
    if list(allowlist) != sorted(set(allowlist)):
        raise ValueError("source allowlist must be unique and sorted")
    files = {relative: sha256_file(root / relative) for relative in allowlist}
    return {
        "schema_version": "TEAM_SOURCE_CLOSURE_V1",
        "files": files,
        "file_count": len(files),
        "composite_sha256": composite_hash(root, allowlist),
    }


def verify_source_closure(root, closure):
    root = Path(root)
    files = closure.get("files")
    if not isinstance(files, dict) or closure.get("file_count") != len(files):
        raise ValueError("source closure is malformed")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    if composite_hash(root, files) != closure.get("composite_sha256"):
        raise ValueError("source composite hash mismatch")
