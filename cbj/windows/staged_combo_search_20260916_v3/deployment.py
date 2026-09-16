"""Frozen combo deployment with verified V2 component-cache import."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from combo_core import active_leaf_ids, apply_bias, evaluate, normalize, probability_sha256
from source_identity import source_identity


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _component_paths(run_root, name, fold):
    root = Path(run_root) / "deployment/components" / name.replace("/", "_") / f"fold_{fold}"
    return root / "probability.npz", root / "COMPLETE.json"


def _import_component(v2_run_root, new_run_root, name, fold, expected_bindings):
    source_npz, source_receipt = _component_paths(v2_run_root, name, fold)
    target_npz, target_receipt = _component_paths(new_run_root, name, fold)
    if not source_npz.is_file() or not source_receipt.is_file():
        return False
    receipt = json.loads(source_receipt.read_text())
    if receipt.get("bindings") != expected_bindings or receipt.get("npz_sha256") != _sha(source_npz):
        raise ValueError(f"V2 component cache lineage mismatch: {name} fold {fold}")
    if target_receipt.is_file():
        current = json.loads(target_receipt.read_text())
        if current != receipt or _sha(target_npz) != receipt["npz_sha256"]:
            raise ValueError("imported component cache changed")
        return True
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = target_npz.with_name(f".{target_npz.name}.partial")
    temporary_receipt = target_receipt.with_name(f".{target_receipt.name}.partial")
    shutil.copyfile(source_npz, temporary_npz); shutil.copyfile(source_receipt, temporary_receipt)
    os.replace(temporary_npz, target_npz); os.replace(temporary_receipt, target_receipt)
    return True


def _write_outputs(run_root, data, test, sample, oof, test_probability, receipt):
    run_root = Path(run_root)
    oof = normalize(oof); test_probability = normalize(test_probability)
    if oof.shape != (len(data["ids"]), len(data["class_order"])):
        raise ValueError("deployment OOF shape mismatch")
    if test_probability.shape != (len(test), len(data["class_order"])):
        raise ValueError("deployment test probability shape mismatch")
    if set(np.unique(data["folds"]).tolist()) != set(range(5)):
        raise ValueError("deployment outer folds differ from 0..4")
    test_ids = np.asarray(test["ID"].astype(str).to_numpy(), dtype="U")
    if not np.array_equal(test_ids, np.asarray(sample["ID"].astype(str).to_numpy(), dtype="U")):
        raise ValueError("test/sample ID order mismatch")
    output = sample.copy(); output["SUBCLASS"] = np.asarray(data["class_order"])[test_probability.argmax(1)]
    submission_path = run_root / "submission_combo_frozen.csv"
    prediction_path = run_root / "predictions.npz"
    if submission_path.exists():
        existing = pd.read_csv(submission_path, encoding="utf-8")
        if not existing.equals(output):
            raise ValueError("existing deployment CSV differs on resume")
    else:
        temporary_csv = submission_path.with_name(f".{submission_path.name}.partial")
        output.to_csv(temporary_csv, index=False); os.replace(temporary_csv, submission_path)
    if prediction_path.exists():
        with np.load(prediction_path, allow_pickle=False) as stored:
            expected = {
                "ids": np.asarray(data["ids"], dtype="U"), "y": data["y_int"],
                "groups": np.asarray(data["groups"], dtype="U"), "folds": data["folds"],
                "class_order": np.asarray(data["class_order"], dtype="U"),
                "oof_probability": oof, "test_ids": test_ids, "test_probability": test_probability,
            }
            if set(stored.files) != set(expected) or any(not np.array_equal(stored[name], value) for name, value in expected.items()):
                raise ValueError("existing deployment predictions differ on resume")
    else:
        temporary_npz = prediction_path.with_name(f".{prediction_path.name}.partial")
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(
                stream, ids=np.asarray(data["ids"], dtype="U"), y=data["y_int"], groups=np.asarray(data["groups"], dtype="U"),
                folds=data["folds"], class_order=np.asarray(data["class_order"], dtype="U"),
                oof_probability=oof, test_ids=test_ids, test_probability=test_probability,
            )
        os.replace(temporary_npz, prediction_path)
    complete = {**receipt, "predictions_sha256": _sha(prediction_path), "submission_sha256": _sha(submission_path)}
    _atomic_json(run_root / "COMPLETE.json", complete)
    return complete


def _verify_provenance(config, v2, prepared, current_v2_source, current_meta_source):
    v2 = Path(v2)
    if current_v2_source != prepared["source_sha256"]:
        raise ValueError("V2 source differs from PREPARED")
    if _sha(v2 / "runtime_config.recovery.windows.json") != prepared["runtime_config_sha256"]:
        raise ValueError("V2 config differs from PREPARED")
    if current_meta_source != config["meta_source_sha256"]:
        raise ValueError("combo meta source differs from frozen config")
    search_receipt = json.loads(Path(config["combo_search_run_root"], "SEARCH_COMPLETE.json").read_text())
    if (search_receipt.get("meta_source_sha256") != current_meta_source
            or search_receipt.get("runtime_config_sha256") != config["_runtime_config_sha256"]
            or search_receipt.get("evidence_sha256") != config["evidence_sha256"]):
        raise ValueError("search provenance differs from deployment config")
    evidence_receipt = json.loads(Path(config["combo_evidence_root"], "EVIDENCE.json").read_text())
    if (_sha(config["evidence_npz"]) != evidence_receipt.get("evidence_sha256")
            or evidence_receipt.get("evidence_sha256") != config["evidence_sha256"]):
        raise ValueError("combo evidence changed before deployment")
    stage_deployment_path = Path(config["combo_evidence_root"]) / "stage_deployment.json"
    if (_sha(stage_deployment_path) != evidence_receipt.get("stage_deployment_sha256")
            or search_receipt.get("stage_deployment_sha256") != evidence_receipt.get("stage_deployment_sha256")):
        raise ValueError("stage deployment bias evidence changed")
    return search_receipt, evidence_receipt, stage_deployment_path


def _verify_search_frozen_binding(config, complete=None):
    search_root = Path(config["combo_search_run_root"])
    search_path = search_root / "SEARCH_COMPLETE.json"
    frozen_path = search_root / "FROZEN_RECIPE.joblib"
    frozen_receipt_path = search_root / "FROZEN_RECIPE.json"
    try:
        search_receipt = json.loads(search_path.read_text())
        frozen_receipt = json.loads(frozen_receipt_path.read_text())
        joblib_sha = _sha(frozen_path)
        receipt_sha = _sha(frozen_receipt_path)
    except Exception as error:
        raise ValueError("completed search/frozen recipe binding changed") from error
    valid = (
        frozen_receipt.get("joblib_sha256") == joblib_sha
        and search_receipt.get("frozen_recipe_joblib_sha256") == joblib_sha
        and search_receipt.get("frozen_recipe_receipt_sha256") == receipt_sha
        and search_receipt.get("best_recipe_id") == frozen_receipt.get("best_recipe_id")
    )
    if complete is not None:
        valid = valid and (
            complete.get("frozen_recipe_sha256") == joblib_sha
            and complete.get("frozen_recipe_receipt_sha256") == receipt_sha
            and complete.get("selected_recipe_id") == frozen_receipt.get("best_recipe_id")
            and complete.get("evidence_sha256") == search_receipt.get("evidence_sha256")
            and complete.get("meta_source_sha256") == search_receipt.get("meta_source_sha256")
            and complete.get("runtime_config_sha256") == search_receipt.get("runtime_config_sha256")
        )
    if not valid:
        raise ValueError("completed search/frozen recipe binding changed")
    return search_receipt, frozen_receipt, frozen_path, frozen_receipt_path


def deploy(config, run_root):
    run_root = Path(run_root); run_root.mkdir(parents=True, exist_ok=True)
    v2 = Path(config["v2_root"]); sys.path.insert(0, str(v2))
    from anchor_pipeline import anchor_model_recipes
    from atomic_state import read_json
    from grade_runner import (
        _component_bindings, _declarations, _fit_anchor_components, _fit_component,
        _load_inference,
    )
    from log_bias import apply_log_bias
    from partition_contract import build_partitions
    from runner import _runner_live
    from support_guard import geometric_pool, guarded_probability
    from workflow import _load_training

    complete_path = run_root / "COMPLETE.json"
    if complete_path.is_file():
        complete = read_json(complete_path)
        _, _, frozen_path, _ = _verify_search_frozen_binding(config, complete)
        if (_sha(run_root / "predictions.npz") != complete.get("predictions_sha256")
                or _sha(run_root / "submission_combo_frozen.csv") != complete.get("submission_sha256")
                or _sha(frozen_path) != complete.get("frozen_recipe_sha256")):
            raise ValueError("completed deployment artifacts changed")
        current_meta_source, _ = source_identity(Path(__file__).parent)
        if (complete.get("meta_source_sha256") != config["meta_source_sha256"]
                or current_meta_source != complete.get("meta_source_sha256")
                or complete.get("runtime_config_sha256") != config["_runtime_config_sha256"]
                or complete.get("evidence_sha256") != config["evidence_sha256"]):
            raise ValueError("completed deployment meta source changed")
        return complete
    if _runner_live(config["v2_run_root"]):
        raise ValueError("V2 worker must be absent before combo deployment")
    prepared = read_json(Path(config["v2_run_root"]) / "PREPARED.json")
    from grade_runner import _source_identity
    current_v2_source, _ = _source_identity(v2)
    current_meta_source, _ = source_identity(Path(__file__).parent)
    search_receipt, evidence_receipt, stage_deployment_path = _verify_provenance(
        config, v2, prepared, current_v2_source, current_meta_source,
    )
    bound_search, frozen_receipt, frozen_path, frozen_receipt_path = _verify_search_frozen_binding(config)
    if bound_search != search_receipt:
        raise ValueError("frozen combo recipe hash mismatch")
    frozen = joblib.load(frozen_path); recipe, nodes = frozen["recipe"], frozen["nodes"]
    stage_deployment = read_json(stage_deployment_path)
    a2_bias = np.asarray(stage_deployment["bias_receipt"]["centered_bias"], dtype=np.float64)
    v2_config = read_json(v2 / "runtime_config.recovery.windows.json")
    data = _load_training(v2_config); test, sample = _load_inference(v2_config, data)
    _, _, declarations = _declarations(v2)
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    stage_recipe = {"root": recipe["stage_graph_root"], "apply_final_guard": False}
    active = set(active_leaf_ids(recipe, nodes))
    active.update(active_leaf_ids(stage_recipe, nodes))
    candidate_ids = sorted(value.split(":", 1)[1] for value in active if value.startswith("candidate:"))
    if any(value not in declarations for value in candidate_ids):
        raise ValueError("frozen combo references an unknown candidate")
    needs_anchors = any(value.startswith("anchor:") for value in active)
    oof = np.zeros((len(data["ids"]), 26), dtype=np.float64); coverage = np.zeros(len(data["ids"]), dtype=np.uint8)
    test_sum = np.zeros((len(test), 26), dtype=np.float64)
    imported = []; fitted = []
    test_ids = test["ID"].astype(str).to_numpy()
    recipes = anchor_model_recipes()
    for outer in partitions["outer"]:
        fold = int(outer["fold"]); train_indices = np.asarray(outer["train_indices"]); valid_indices = np.asarray(outer["valid_indices"])
        missing_anchors = []
        for name in (("ANCHOR-H1", "H1"), ("ANCHOR-C10", "C10")) if needs_anchors else ():
            component, recipe_name = name
            bindings = _component_bindings(
                component, recipes[recipe_name]["config_hash"], fold, data,
                train_indices, valid_indices, test_ids, prepared["source_sha256"],
            )
            if _import_component(config["v2_run_root"], run_root, component, fold, bindings):
                imported.append([component, fold])
            else:
                missing_anchors.append(component)
        for candidate_id in candidate_ids:
            declaration = declarations[candidate_id]
            bindings = _component_bindings(
                candidate_id, declaration["config_hash"], fold, data,
                train_indices, valid_indices, test_ids, prepared["source_sha256"],
            )
            if _import_component(config["v2_run_root"], run_root, candidate_id, fold, bindings):
                imported.append([candidate_id, fold])
        leaves_valid = {}; leaves_test = {}
        if needs_anchors:
            anchors = _fit_anchor_components(
                v2_config, run_root, data, test, fold, train_indices, valid_indices,
                prepared["source_sha256"],
            )
            fitted.extend([[component, fold] for component in missing_anchors])
            h1_valid, h1_test = anchors["H1"]; c10_valid, c10_test = anchors["C10"]
            guard_all = np.asarray(anchors["guard"]); boundary = len(valid_indices)
            leaves_valid["anchor:A1_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_valid, c10_valid), h1_valid, guard_all[:boundary],
            )
            leaves_test["anchor:A1_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_test, c10_test), h1_test, guard_all[boundary:],
            )
            leaves_valid["anchor:A2_C10_BIAS_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_valid, apply_log_bias(c10_valid, a2_bias)), h1_valid, guard_all[:boundary],
            )
            leaves_test["anchor:A2_C10_BIAS_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_test, apply_log_bias(c10_test, a2_bias)), h1_test, guard_all[boundary:],
            )
            leaves_valid["mask:guard"] = guard_all[:boundary]; leaves_test["mask:guard"] = guard_all[boundary:]
        else:
            raise ValueError("stage baseline deployment requires anchors")
        for candidate_id in candidate_ids:
            before = _component_paths(run_root, candidate_id, fold)[1].is_file()
            valid_probability, test_probability = _fit_component(
                v2_config, run_root, data, test, declarations[candidate_id], fold,
                train_indices, valid_indices, prepared["source_sha256"],
            )
            if not before: fitted.append([candidate_id, fold])
            leaves_valid[f"candidate:{candidate_id}"] = valid_probability
            leaves_test[f"candidate:{candidate_id}"] = test_probability
        stage_valid = evaluate(stage_recipe, nodes, leaves_valid, mode="deployment")
        stage_test = evaluate(stage_recipe, nodes, leaves_test, mode="deployment")
        leaves_valid["stage:latest"] = stage_valid; leaves_test["stage:latest"] = stage_test
        fold_valid = evaluate(recipe, nodes, leaves_valid, mode="deployment")
        fold_test = evaluate(recipe, nodes, leaves_test, mode="deployment")
        oof[valid_indices] = fold_valid; coverage[valid_indices] += 1; test_sum += fold_test
    if not np.all(coverage == 1): raise ValueError("combo deployment OOF coverage mismatch")
    test_probability = normalize(test_sum / 5)
    full_oof = float(f1_score(data["y_int"], oof.argmax(1), labels=np.arange(26), average="macro", zero_division=0))
    receipt = {
        "schema_version": "FROZEN_COMBO_DEPLOYMENT_COMPLETE_V1", "status": "COMPLETE",
        "base_source_sha256": prepared["source_sha256"],
        "meta_source_sha256": config["meta_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "evidence_sha256": config["evidence_sha256"],
        "frozen_recipe_sha256": frozen_receipt["joblib_sha256"],
        "frozen_recipe_receipt_sha256": _sha(frozen_receipt_path),
        "selected_recipe_id": frozen_receipt["best_recipe_id"],
        "adaptive_full_oof_macro_f1": full_oof,
        "imported_component_folds": imported, "newly_fitted_component_folds": fitted,
        "selected_components": candidate_ids, "test_used_for_selection": False,
        "full_oof_used_for_recipe_selection": False, "website_submitted": False,
    }
    return _write_outputs(run_root, data, test, sample, oof, test_probability, receipt)
