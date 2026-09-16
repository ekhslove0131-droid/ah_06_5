"""Read-only preflight and explicit detached launch CLI for Stack7 meta search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

try:
    from .artifact_store import load_full_bank, load_inner_bank, sequence_sha256
    from .baseline import load_baseline_inner
    from .meta_core import declarations
    from .meta_deployment import verify_frozen_selection
    from .source_identity import file_sha256, source_identity
except ImportError:  # supported standalone package execution
    from artifact_store import load_full_bank, load_inner_bank, sequence_sha256
    from baseline import load_baseline_inner
    from meta_core import declarations
    from meta_deployment import verify_frozen_selection
    from source_identity import file_sha256, source_identity


APPROVED_RUNNER_SOURCE_SHA256 = "b56d607e5da1edf3f0e4c97e426182896f094ba35faf038735ceabe03b343b52"
PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = {
    "heads": 54,
    "inner_fits": 162,
    "mixtures": 540,
    "controls": 1,
    "score_rows": 541,
}
EXPECTED_AXES = {
    "representations": ["log_probability_confidence"],
    "depth": [3, 5, 7],
    "iterations": [200, 600, 1200],
    "l2_leaf_reg": [3, 30],
    "weightings": ["none", "sqrt_inverse", "balanced"],
    "inner_folds": [0, 1, 2],
    "alphas": [0.1, 0.25, 0.5, 0.75, 1.0],
    "poolings": ["arithmetic", "geometric"],
}


def _read_json(path, label="JSON artifact"):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected an object")
    return value


def load_runtime_config(path):
    path = Path(path)
    config = _read_json(path, "runtime config")
    config["_runtime_config_sha256"] = file_sha256(path)
    config["_runtime_config_path"] = str(path.resolve())
    return config


def _validate_grid_manifest(path):
    manifest = _read_json(path, "grid manifest")
    if (manifest.get("schema_version") != "STACK7_META_GRID_V1"
            or manifest.get("declarations") != declarations()
            or manifest.get("axes") != EXPECTED_AXES
            or manifest.get("counts") != EXPECTED_COUNTS
            or manifest.get("ordering") != "declarations_then_alpha_then_pooling_with_baseline_control_first"
            or manifest.get("alpha_one_geometric_is_alias") is not True):
        raise ValueError("grid manifest does not declare the fixed 54/162/541 bundle")
    return manifest


def verify_package(config, config_path, package_root=PACKAGE_ROOT):
    """Verify source, manifests, and external config binding without writing."""
    root = Path(package_root)
    config_path = Path(config_path)
    source_sha, rows = source_identity(root)
    if config.get("execution") != "GPU_REQUIRED_NO_CPU_FALLBACK":
        raise ValueError("runtime execution must require CatBoost GPU without CPU fallback")
    if config.get("meta_source_sha256") != source_sha:
        raise ValueError("meta source identity changed")

    source_manifest_path = Path(config.get("source_manifest_json", ""))
    if (not source_manifest_path.is_file()
            or file_sha256(source_manifest_path) != config.get("source_manifest_sha256")):
        raise ValueError("source manifest hash mismatch")
    source_manifest = _read_json(source_manifest_path, "source manifest")
    if (source_manifest.get("schema_version") != "STACK7_META_SOURCE_MANIFEST_V1"
            or source_manifest.get("source_sha256") != source_sha
            or source_manifest.get("files") != rows):
        raise ValueError("source manifest binding mismatch")

    grid_path = Path(config.get("grid_manifest_json", ""))
    if not grid_path.is_file() or file_sha256(grid_path) != config.get("grid_manifest_sha256"):
        raise ValueError("grid manifest hash mismatch")
    _validate_grid_manifest(grid_path)

    prepared_path = Path(config.get("prepared_source_json", ""))
    prepared = _read_json(prepared_path, "prepared source receipt")
    if (prepared.get("schema_version") != "STACK7_META_PREPARED_SOURCE_V1"
            or prepared.get("meta_source_sha256") != source_sha
            or prepared.get("source_files") != rows
            or prepared.get("source_file_count") != len(rows)
            or prepared.get("source_manifest_sha256") != file_sha256(source_manifest_path)
            or prepared.get("grid_manifest_sha256") != file_sha256(grid_path)
            or prepared.get("runtime_config_windows_sha256") != file_sha256(config_path)
            or prepared.get("declared_counts") != EXPECTED_COUNTS):
        raise ValueError("prepared source/runtime config binding mismatch")
    return {
        "status": "PACKAGE_VERIFIED",
        "meta_source_sha256": source_sha,
        "runtime_config_sha256": file_sha256(config_path),
        "source_file_count": len(rows),
    }


def validate_search_inputs(config):
    """Validate only the inner bank and frozen inner baseline; never full/test."""
    bank = load_inner_bank(config)
    baseline, guard, lineage = load_baseline_inner(config)
    rows = len(bank["inner_ids"])
    classes = len(bank["class_order"])
    expected_models = list(config.get("expected_model_ids", []))
    if bank["model_ids"].astype(str).tolist() != expected_models:
        raise ValueError("inner bank model order mismatch")
    if baseline.shape != (rows, classes) or np.asarray(guard).shape != (rows,):
        raise ValueError("inner baseline dimensions mismatch")
    if (not np.isfinite(baseline).all() or (baseline < 0).any()
            or not np.allclose(baseline.sum(1), 1.0, rtol=0.0, atol=1e-8)):
        raise ValueError("inner baseline probability is invalid")
    bindings = {
        "ids_sha256": sequence_sha256(bank["inner_ids"]),
        "y_sha256": sequence_sha256(bank["inner_y"]),
        "groups_sha256": sequence_sha256(bank["inner_groups"]),
        "outer_folds_sha256": sequence_sha256(bank["inner_outer_folds"]),
        "inner_folds_sha256": sequence_sha256(bank["inner_folds"]),
        "class_order_sha256": sequence_sha256(bank["class_order"]),
    }
    for key, expected in bindings.items():
        if lineage.get(key) != expected:
            raise ValueError(f"inner bank/baseline binding mismatch: {key}")
    return {
        "status": "INNER_INPUTS_VERIFIED",
        "rows": rows,
        "classes": classes,
        "models": len(expected_models),
        "test_read": False,
        "full_bank_read": False,
    }


def resolve_deployment_config(config):
    """Bind future search outputs in memory without changing frozen config bytes."""
    resolved = copy.deepcopy(config)
    root = Path(config["meta_search_run_root"])
    selection_path = root / "FROZEN_SELECTION.json"
    complete_path = root / "SEARCH_COMPLETE.json"
    if not selection_path.is_file() or not complete_path.is_file():
        raise ValueError("meta search is not complete")
    resolved["meta_search_selection_sha256"] = file_sha256(selection_path)
    resolved["meta_search_complete_sha256"] = file_sha256(complete_path)
    return resolved


def validate_deployment_freeze(config):
    selection, complete = verify_frozen_selection(config)
    if selection.get("deployment_required") is False:
        return {
            "status": "INCUMBENT_NO_DEPLOYMENT_READY",
            "deployment_required": False,
            "selected_kind": selection.get("selected_kind"),
            "test_read": False,
            "full_bank_read": False,
        }
    return {
        "status": "FROZEN_META_DEPLOYMENT_READY",
        "deployment_required": True,
        "selected_kind": selection.get("selected_kind"),
        "search_status": complete.get("status"),
        "test_read": False,
        "full_bank_read": False,
    }


def ensure_no_active_workers(config, runner_live):
    roots = config.get("known_worker_run_roots")
    if not isinstance(roots, list) or not roots or len(roots) != len(set(roots)):
        raise ValueError("known worker run roots must be a non-empty unique list")
    active = [str(root) for root in roots if runner_live(Path(root))]
    if active:
        raise ValueError("active worker blocks launch: " + ",".join(active))
    return roots


def _runner_source_identity(root):
    """Reproduce the approved recovery source identity before importing runner.py."""
    root = Path(root)
    names = sorted(
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.startswith("._")
        and path.suffix in {".py", ".json"}
        and not str(path.relative_to(root)).startswith(("runs/", "artifacts/"))
        and path.name not in {"runtime_config.windows.json", "LOCAL_ACCEPTANCE.json"}
    )
    rows = [{"path": name, "sha256": file_sha256(root / name)} for name in names]
    payload = (json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def load_verified_runner(config):
    root = Path(config.get("v2_root", ""))
    actual = _runner_source_identity(root)
    if actual != APPROVED_RUNNER_SOURCE_SHA256:
        raise ValueError("approved detached runner source identity changed")
    runner_path = root / "runner.py"
    if not runner_path.is_file():
        raise ValueError("approved detached runner is missing")
    unique_name = f"_stack7_approved_runner_{actual}"
    spec = importlib.util.spec_from_file_location(unique_name, runner_path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to load approved detached runner")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)
    if not callable(getattr(module, "launch_detached", None)) or not callable(getattr(module, "_runner_live", None)):
        raise ValueError("approved detached runner API is incomplete")
    return module


def preflight(config, config_path, package_root=PACKAGE_ROOT, *, phase="search"):
    package = verify_package(config, config_path, package_root)
    runner = load_verified_runner(config)
    ensure_no_active_workers(config, runner._runner_live)
    if phase == "search":
        inputs = validate_search_inputs(config)
    elif phase == "deploy":
        resolved = resolve_deployment_config(config)
        inputs = validate_deployment_freeze(resolved)
    else:
        raise ValueError(f"unknown preflight phase: {phase}")
    return {"status": "READY", "phase": phase, "package": package, "inputs": inputs}


def status(config, config_path, package_root=PACKAGE_ROOT):
    package = verify_package(config, config_path, package_root)
    inputs = validate_search_inputs(config)
    runner = load_verified_runner(config)
    result = {"status": "READ_ONLY_STATUS", "package": package, "inputs": inputs, "runs": {}}
    for name, root_text in config.get("status_run_roots", {}).items():
        root = Path(root_text)
        complete_candidates = ("SEARCH_COMPLETE.json", "COMPLETE.json", "NO_DEPLOYMENT.json")
        complete = next((root / name for name in complete_candidates if (root / name).is_file()), None)
        failure = root / "FAILURE.json"
        result["runs"][name] = {
            "runner_live": runner._runner_live(root),
            "complete": _read_json(complete) if complete else None,
            "failure": _read_json(failure) if failure.is_file() else None,
        }
    return result


def _worker(command, config_path, package_root):
    config = load_runtime_config(config_path)
    verify_package(config, config_path, package_root)
    if command == "search":
        validate_search_inputs(config)
        try:
            from .meta_search import run_search
        except ImportError:
            from meta_search import run_search
        return run_search(config, config["meta_search_run_root"])
    resolved = resolve_deployment_config(config)
    validate_deployment_freeze(resolved)
    try:
        from .meta_deployment import run_deployment
    except ImportError:
        from meta_deployment import run_deployment
    return run_deployment(resolved, resolved["meta_deployment_run_root"])


def build_parser():
    parser = argparse.ArgumentParser(prog="cbj-stack7-meta")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--phase", choices=("search", "deploy"), default="search")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--config", required=True)
    for command in ("search", "deploy"):
        current = subparsers.add_parser(command)
        current.add_argument("--config", required=True)
        current.add_argument("--start", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in {"search", "deploy"} and not args.start:
        print(json.dumps({"status": "START_FLAG_REQUIRED"}, sort_keys=True))
        return 2
    config_path = Path(args.config)
    try:
        config = load_runtime_config(config_path)
        if args.command == "status":
            result = status(config, config_path, PACKAGE_ROOT)
        elif args.command == "preflight":
            result = preflight(config, config_path, PACKAGE_ROOT, phase=args.phase)
        else:
            preflight(config, config_path, PACKAGE_ROOT, phase=args.command)
            runner = load_verified_runner(config)
            ensure_no_active_workers(config, runner._runner_live)
            run_root = config["meta_search_run_root" if args.command == "search" else "meta_deployment_run_root"]
            result = runner.launch_detached(
                run_root,
                lambda: _worker(args.command, config_path, PACKAGE_ROOT),
                start=True,
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        if args.command in {"search", "deploy"}:
            return 0 if result.get("status") == "STARTED" else 3
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
