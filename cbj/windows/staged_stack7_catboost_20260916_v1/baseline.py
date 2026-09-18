"""Verified adapter for the pinned V3 baseline graph."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

try:
    from .artifact_store import canonical_sha256, file_sha256, probability_sha256
except ImportError:  # supported direct script import
    from artifact_store import canonical_sha256, file_sha256, probability_sha256


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid baseline JSON artifact: {path}") from exc


def _source_closure(root):
    root = Path(root)
    allowlist = root / "SOURCE_ALLOWLIST.txt"
    names = [line.strip() for line in allowlist.read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)) or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("V3 source allowlist is invalid")
    rows = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise ValueError(f"V3 source closure is incomplete: {name}")
        rows.append({"path": name, "sha256": file_sha256(path)})
    raw = (json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest(), rows


def _load_verified_evaluator(source_root, expected_source):
    actual_source, rows = _source_closure(source_root)
    if actual_source != expected_source:
        raise ValueError("V3 source identity changed")
    row_map = {row["path"]: row["sha256"] for row in rows}
    if "combo_core.py" not in row_map:
        raise ValueError("V3 source closure does not bind combo_core.py")
    path = Path(source_root) / "combo_core.py"
    # Source closure and the module byte hash are checked before execution.
    unique = f"_stack7_verified_combo_core_{row_map['combo_core.py']}"
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to construct verified V3 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (not callable(getattr(module, "evaluate", None))
            or not callable(getattr(module, "normalize", None))
            or not callable(getattr(module, "probability_sha256", None))):
        raise ValueError("verified V3 evaluator API is incomplete")
    return module, rows


def _verified_frozen_context(config):
    """Load verified V3 evaluator and frozen graph; reusable by Task4 fold replay."""
    required = {
        "v3_source_root", "v3_runtime_config_path", "v3_search_root", "v3_source_sha256",
        "v3_runtime_config_sha256", "v3_frozen_recipe_sha256",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"baseline config missing keys: {sorted(missing)}")
    runtime_path = Path(config["v3_runtime_config_path"])
    if file_sha256(runtime_path) != config["v3_runtime_config_sha256"]:
        raise ValueError("V3 runtime config hash mismatch")
    runtime = _read(runtime_path)
    if runtime.get("meta_source_sha256") != config["v3_source_sha256"]:
        raise ValueError("V3 runtime/source binding mismatch")
    evaluator, source_rows = _load_verified_evaluator(config["v3_source_root"], config["v3_source_sha256"])
    root = Path(config["v3_search_root"])
    frozen_path = root / "FROZEN_RECIPE.joblib"
    frozen_receipt_path = root / "FROZEN_RECIPE.json"
    search_path = root / "SEARCH_COMPLETE.json"
    if file_sha256(frozen_path) != config["v3_frozen_recipe_sha256"]:
        raise ValueError("V3 frozen recipe hash mismatch")
    frozen_receipt = _read(frozen_receipt_path)
    search = _read(search_path)
    recipe_id = frozen_receipt.get("best_recipe_id")
    if (search.get("status") != "COMPLETE" or not recipe_id
            or search.get("best_recipe_id") != recipe_id
            or frozen_receipt.get("joblib_sha256") != config["v3_frozen_recipe_sha256"]
            or search.get("frozen_recipe_joblib_sha256") != config["v3_frozen_recipe_sha256"]
            or search.get("frozen_recipe_receipt_sha256") != file_sha256(frozen_receipt_path)
            or search.get("meta_source_sha256") != config["v3_source_sha256"]
            or search.get("runtime_config_sha256") != config["v3_runtime_config_sha256"]
            or search.get("test_read") is not False or search.get("full_oof_used") is not False):
        raise ValueError("V3 frozen receipt/search/source/config binding changed")
    # Frozen bytes and every producer receipt binding are checked before unpickling.
    frozen = joblib.load(frozen_path)
    if set(frozen) != {"recipe", "nodes"} or frozen.get("recipe", {}).get("root") != recipe_id:
        raise ValueError("V3 frozen graph payload is invalid")
    return {
        "evaluator": evaluator, "frozen": frozen, "frozen_receipt": frozen_receipt,
        "search": search, "source_rows": source_rows,
        "frozen_path": frozen_path, "frozen_receipt_path": frozen_receipt_path,
    }


def _load_evidence(config):
    required = {"evidence_npz", "evidence_receipt_json", "evidence_sha256", "expected_inner_rows", "expected_classes"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"baseline evidence config missing keys: {sorted(missing)}")
    path = Path(config["evidence_npz"])
    if file_sha256(path) != config["evidence_sha256"]:
        raise ValueError("baseline evidence hash mismatch")
    receipt = _read(config["evidence_receipt_json"])
    if receipt.get("evidence_sha256") != config["evidence_sha256"]:
        raise ValueError("baseline evidence receipt mismatch")
    stage_path = path.parent / "stage_deployment.json"
    if not stage_path.is_file() or receipt.get("stage_deployment_sha256") != file_sha256(stage_path):
        raise ValueError("baseline stage deployment binding mismatch")
    required_members = (
        "ids", "y", "groups", "outer_folds", "inner_folds", "class_order", "guard",
        "candidate_ids", "candidate_probabilities", "anchor_ids", "anchor_probabilities", "stage_probability",
    )
    try:
        with np.load(path, allow_pickle=False) as stored:
            if not set(required_members).issubset(set(stored.files)):
                raise ValueError("baseline evidence fields missing")
            evidence = {name: np.asarray(stored[name]) for name in required_members}
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("baseline evidence members are invalid") from exc
    for name in ("ids", "y", "groups", "outer_folds", "inner_folds", "class_order"):
        if receipt.get(f"{name}_sha256") != canonical_sha256(evidence[name].tolist()):
            raise ValueError(f"baseline evidence {name} identity mismatch")
    rows, classes = int(config["expected_inner_rows"]), int(config["expected_classes"])
    if (evidence["ids"].shape != (rows,) or evidence["y"].shape != (rows,)
            or evidence["groups"].shape != (rows,) or evidence["outer_folds"].shape != (rows,)
            or evidence["inner_folds"].shape != (rows,) or evidence["class_order"].shape != (classes,)
            or evidence["guard"].dtype != np.bool_ or evidence["guard"].shape != (rows,)
            or evidence["stage_probability"].shape != (rows, classes)):
        raise ValueError("baseline evidence dimensions are invalid")
    if set(np.unique(evidence["inner_folds"]).tolist()) != {0, 1, 2}:
        raise ValueError("baseline evidence inner folds are invalid")
    expected_candidates = config.get("expected_evidence_candidates")
    if expected_candidates is not None and len(evidence["candidate_ids"]) != int(expected_candidates):
        raise ValueError("baseline candidate evidence count changed")
    return evidence, receipt


def _leaves(evidence, evaluator):
    leaves = {
        "stage:latest": evaluator.normalize(evidence["stage_probability"]),
        "mask:guard": evidence["guard"],
    }
    for index, candidate_id in enumerate(evidence["candidate_ids"].astype(str).tolist()):
        leaves[f"candidate:{candidate_id}"] = evaluator.normalize(evidence["candidate_probabilities"][index])
    for index, anchor_id in enumerate(evidence["anchor_ids"].astype(str).tolist()):
        leaves[f"anchor:{anchor_id}"] = evaluator.normalize(evidence["anchor_probabilities"][index])
    return leaves


def load_baseline_inner(config: dict) -> tuple:
    """Replay the pinned V3 inner baseline after complete provenance checks."""
    context = _verified_frozen_context(config)
    evidence, evidence_receipt = _load_evidence(config)
    search = context["search"]
    if (search.get("evidence_sha256") != config["evidence_sha256"]
            or search.get("stage_deployment_sha256") != evidence_receipt.get("stage_deployment_sha256")):
        raise ValueError("V3 baseline/evidence binding changed")
    frozen = context["frozen"]
    probability = np.asarray(context["evaluator"].evaluate(
        frozen["recipe"], frozen["nodes"], _leaves(evidence, context["evaluator"]),
        mode="search", fold_ids=evidence["inner_folds"],
    ), dtype=np.float64)
    rows, classes = int(config["expected_inner_rows"]), int(config["expected_classes"])
    if (probability.shape != (rows, classes) or not np.isfinite(probability).all()
            or (probability < 0).any() or not np.allclose(probability.sum(1), 1.0, rtol=0.0, atol=1e-8)):
        raise ValueError("V3 baseline probability is invalid")
    score = float(f1_score(
        evidence["y"].astype(np.int64), probability.argmax(1), labels=np.arange(classes),
        average="macro", zero_division=0,
    ))
    expected_score = float(config["expected_baseline_inner_macro_f1"])
    if abs(score - expected_score) > 1e-15:
        raise ValueError("V3 baseline inner score does not reproduce")
    probability_hash = probability_sha256(probability)
    expected_probability_hash = context["frozen_receipt"].get("frozen_reproduced_probability_sha256")
    if expected_probability_hash and expected_probability_hash != probability_hash:
        raise ValueError("V3 baseline inner probability does not reproduce")
    lineage = {
        "baseline_recipe_id": context["frozen_receipt"]["best_recipe_id"],
        "baseline_inner_macro_f1": score,
        "baseline_probability_sha256": probability_hash,
        "v3_frozen_recipe_sha256": config["v3_frozen_recipe_sha256"],
        "v3_frozen_recipe_receipt_sha256": file_sha256(context["frozen_receipt_path"]),
        "v3_source_sha256": config["v3_source_sha256"],
        "v3_runtime_config_sha256": config["v3_runtime_config_sha256"],
        "evidence_sha256": config["evidence_sha256"],
        "ids_sha256": evidence_receipt["ids_sha256"],
        "y_sha256": evidence_receipt["y_sha256"],
        "groups_sha256": evidence_receipt["groups_sha256"],
        "outer_folds_sha256": evidence_receipt["outer_folds_sha256"],
        "inner_folds_sha256": evidence_receipt["inner_folds_sha256"],
        "class_order_sha256": evidence_receipt["class_order_sha256"],
        "stage_deployment_sha256": evidence_receipt["stage_deployment_sha256"],
    }
    return probability, evidence["guard"].copy(), lineage


def _original_geometric_pool(left, right):
    """Original support_guard geometry: normalize, log floor 1e-300, equal weights."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 2:
        raise ValueError("V3 anchor probability shapes are misaligned")
    if (not np.isfinite(left).all() or not np.isfinite(right).all()
            or (left < 0).any() or (right < 0).any()
            or np.any(left.sum(1) <= 0) or np.any(right.sum(1) <= 0)):
        raise ValueError("V3 anchor probability values are invalid")
    left = left / left.sum(1, keepdims=True)
    right = right / right.sum(1, keepdims=True)
    logits = (np.log(np.maximum(left, 1e-300)) + np.log(np.maximum(right, 1e-300))) / 2.0
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def _original_log_bias(probability, bias):
    probability = np.asarray(probability, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if probability.ndim != 2 or bias.shape != (probability.shape[1],):
        raise ValueError("V3 A2 bias is misaligned")
    probability = probability / probability.sum(1, keepdims=True)
    logits = np.log(np.maximum(probability, 1e-300)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def replay_v3_fold(config, raw_probability, guard):
    """Replay one row set of a frozen V3 fold from seven verified caches."""
    context = _verified_frozen_context(config)
    stage_path = Path(config.get("v3_stage_deployment_json", ""))
    expected_stage = config.get("v3_stage_deployment_sha256")
    if not stage_path.is_file() or not expected_stage or file_sha256(stage_path) != expected_stage:
        raise ValueError("V3 stage deployment bias binding changed")
    stage = _read(stage_path)
    bias = np.asarray(stage.get("bias_receipt", {}).get("centered_bias"), dtype=np.float64)
    required = (
        "ANCHOR-H1", "ANCHOR-C10", "OLD-G3-0014", "OLD-G3-0018",
        "TEAM-T2-0005", "TEAM-T2-0007", "TEAM-T4-0007",
    )
    if set(raw_probability) != set(required):
        raise ValueError("V3 fold raw component order/membership changed")

    def leaves(raw, guard):
        guard = np.asarray(guard)
        h1 = np.asarray(raw["ANCHOR-H1"], dtype=np.float64)
        c10 = np.asarray(raw["ANCHOR-C10"], dtype=np.float64)
        if guard.dtype != np.bool_ or guard.shape != (len(h1),):
            raise ValueError("V3 fold guard is misaligned")
        a1 = _original_geometric_pool(h1, c10)
        a2 = _original_geometric_pool(h1, _original_log_bias(c10, bias))
        normalized_h1 = h1 / h1.sum(1, keepdims=True)
        a1[guard] = normalized_h1[guard]
        a2[guard] = normalized_h1[guard]
        value = {
            "anchor:A1_GEOMETRIC_GUARD": a1,
            "anchor:A2_C10_BIAS_GEOMETRIC_GUARD": a2,
            "mask:guard": guard,
        }
        for name in required[2:]:
            value[f"candidate:{name}"] = np.asarray(raw[name], dtype=np.float64)
        return value

    frozen = context["frozen"]
    stage_recipe = {"root": frozen["recipe"]["stage_graph_root"], "apply_final_guard": False}
    fold_leaves = leaves(raw_probability, guard)
    fold_leaves["stage:latest"] = context["evaluator"].evaluate(
        stage_recipe, frozen["nodes"], fold_leaves, mode="deployment",
    )
    return np.asarray(context["evaluator"].evaluate(
        frozen["recipe"], frozen["nodes"], fold_leaves, mode="deployment",
    ), dtype=np.float64)
