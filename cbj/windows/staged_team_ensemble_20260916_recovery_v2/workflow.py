"""Real nested team-ensemble workflow plus a reduced synthetic acceptance path."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anchor_pipeline import (
    anchor_model_recipes, build_anchor_oof, prepare_anchor_features,
    prepare_c10_features,
)
from atomic_state import append_jsonl, atomic_write_json, read_json
from base_bank import run_base_bank
from base_receipts import canonical_sha256, sequence_sha256
from cache_store import ImmutableProbabilityCache, make_cache_key
from contracts import CLASS_ORDER, TRAIN_SHA256
from eligibility_audit import audit_admission, slot_for
from feature_transforms import FeatureContext, fit_transform_candidate
from inventory import canonical_bytes
from log_bias import apply_log_bias, fit_c10_log_bias
from meta_rounds import run_meta_round
from metrics import assemble_final_oof, load_final_oof, milestone_status, score_final_oof
from model_adapters import effective_recipe, fit_model, predict_checked
from neural_candidates import prepare_neural_candidate
from numeric_views import build_numeric41
from old_candidates import fit_transform_old_candidate
from outer_procedure import select_inner_recipe, select_then_predict_outer
from partition_contract import build_partitions
from source_closure import sha256_file, verify_source_closure
from support_guard import geometric_pool, guarded_probability


def _write_json_once(path, value):
    path = Path(path)
    data = canonical_bytes(value)
    if path.is_file():
        if path.read_bytes() != data:
            raise ValueError(f"immutable JSON collision: {path}")
        return
    atomic_write_json(path, value)


def _save_joblib_once(path, value):
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)
    return value


def _load_training(config):
    train_path = Path(config["train_csv"])
    reference_path = Path(config["reference_oof"])
    if train_path.name.lower() != "train.csv":
        raise ValueError("only train.csv is admitted as tabular runtime input")
    if sha256_file(train_path) != config.get("train_sha256", TRAIN_SHA256):
        raise ValueError("train.csv hash mismatch")
    if sha256_file(reference_path) != config["reference_oof_sha256"]:
        raise ValueError("reference identity artifact hash mismatch")
    frame = pd.read_csv(train_path, encoding="utf-8")
    if list(frame.columns[:2]) != ["ID", "SUBCLASS"]:
        raise ValueError("training schema must begin with ID,SUBCLASS")
    with np.load(reference_path, allow_pickle=False) as stored:
        required = ("ids", "y", "groups", "folds", "class_order")
        reference = {key: np.asarray(stored[key]) for key in required}
    if reference["class_order"].tolist() != CLASS_ORDER:
        raise ValueError("reference class order mismatch")
    if not np.array_equal(frame["ID"].to_numpy(dtype=str), reference["ids"]):
        raise ValueError("train/reference ID order mismatch")
    labels = frame["SUBCLASS"].to_numpy(dtype=str)
    expected_labels = reference["class_order"][reference["y"]]
    if not np.array_equal(labels, expected_labels):
        raise ValueError("train/reference labels mismatch")
    gene_cols = frame.columns[2:].tolist()
    return {
        "frame": frame,
        "genes": gene_cols,
        "ids": reference["ids"],
        "y_int": reference["y"].astype(np.int64),
        "y_label": labels,
        "groups": reference["groups"],
        "folds": reference["folds"],
        "class_order": reference["class_order"].tolist(),
    }


def _context(data, fit_indices, valid_indices):
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    train = data["frame"].iloc[fit_indices][data["genes"]].reset_index(drop=True)
    valid = data["frame"].iloc[valid_indices][data["genes"]].reset_index(drop=True)
    numeric = build_numeric41(train, valid, data["genes"], data["groups"][fit_indices])
    context = FeatureContext(
        gene_cols=data["genes"], train_ids=data["ids"][fit_indices],
        valid_ids=data["ids"][valid_indices], train_groups=data["groups"][fit_indices],
        train_y=data["y_label"][fit_indices], class_order=data["class_order"],
        numeric41_train=numeric["train"], numeric41_valid=numeric["valid"],
    )
    return train, valid, context


def _prepare_candidate(declaration, train, valid, context):
    family = declaration["family"]
    if family in {"G1", "G2", "G3", "G4"}:
        return fit_transform_old_candidate(declaration, train, valid, context)
    if family in {"G5", "G6"}:
        return prepare_neural_candidate(declaration, train, valid, context)
    return fit_transform_candidate(declaration, train, valid, context)


def _feature_order_hash(features):
    names = getattr(features, "feature_names", None)
    if names is not None:
        return canonical_sha256(list(names))
    return canonical_sha256(features.state["fitted"])


def _fit_prepared(recipe, features, context):
    artifact = fit_model(recipe, features, context.train_y, None, context.class_order)
    prediction_input = features.valid
    if (recipe.get("transformer") or {}).get("kind") == "AE64":
        if features.deferred_numeric_valid is None:
            raise ValueError("AE64 prediction requires deferred numeric41")
        prediction_input = {
            "ae_input": features.valid,
            "numeric41": features.deferred_numeric_valid,
        }
    return predict_checked(artifact, prediction_input, context.class_order)


def _cached_prepared_probability(cache, namespace, recipe, features, context, source_sha, partition_sha):
    bindings = {
        "input_sha256": TRAIN_SHA256,
        "parser_sha256": sha256_file(Path(__file__).parent / "vendor/mcmp_repr.py"),
        "source_sha256": source_sha,
        "declaration_sha256": recipe["config_hash"],
        "fit_ids_sha256": sequence_sha256(context.train_ids),
        "fit_y_sha256": sequence_sha256(context.train_y),
        "fit_groups_sha256": sequence_sha256(context.train_groups),
        "partition_sha256": canonical_sha256([namespace, partition_sha]),
        "feature_order_sha256": _feature_order_hash(features),
        "parameters_sha256": canonical_sha256(recipe),
    }
    key = make_cache_key(bindings, supervised=True)
    if cache.artifact_path(key).is_dir():
        probability, _ = cache.load_or_adopt(key, bindings)
        return probability
    probability = _fit_prepared(recipe, features, context)
    probability, _ = cache.write(key, probability, bindings)
    return probability


def _fit_candidate_boundary(declaration, data, fit_indices, valid_indices, cache, source_sha, partition_sha):
    train, valid, context = _context(data, fit_indices, valid_indices)
    features = _prepare_candidate(declaration, train, valid, context)
    probability = _cached_prepared_probability(
        cache, declaration["candidate_id"], effective_recipe(declaration), features,
        context, source_sha, partition_sha,
    )
    return probability, {"feature_order_sha256": _feature_order_hash(features)}


def _fit_c10_boundary(data, fit_indices, valid_indices, cache, source_sha, partition_sha, namespace):
    train, valid, context = _context(data, fit_indices, valid_indices)
    features, _, _ = prepare_c10_features(train, valid, context)
    probability = _cached_prepared_probability(
        cache, namespace, anchor_model_recipes()["C10"], features, context,
        source_sha, partition_sha,
    )
    return probability


def _fit_anchor_boundary(data, fit_indices, valid_indices, cache, source_sha, partition_sha, namespace):
    train, valid, context = _context(data, fit_indices, valid_indices)
    features = prepare_anchor_features(train, valid, context)
    recipes = anchor_model_recipes()
    probabilities = {}
    for name in ("H1", "C10"):
        probabilities[name] = _cached_prepared_probability(
            cache, f"{namespace}-{name}", recipes[name], features[name], context,
            source_sha, canonical_sha256([partition_sha, name]),
        )
    return {
        "probabilities": probabilities,
        "guard_mask": features["guard_mask"],
        "guard_receipt": features["guard_receipt"],
    }


def _inner_work(outer, data, declarations, run_root, source_sha):
    outer_indices = np.asarray(outer["train_indices"], dtype=np.int64)
    local_by_global = {int(value): index for index, value in enumerate(outer_indices)}
    inner_folds = np.empty(len(outer_indices), dtype=np.int16)
    base_cache = ImmutableProbabilityCache(Path(run_root) / "cache" / f"outer_{outer['fold']}" / "base")
    anchor_cache = ImmutableProbabilityCache(Path(run_root) / "cache" / f"outer_{outer['fold']}" / "anchors")
    anchor_oof = {
        "A1_GEOMETRIC_GUARD": np.zeros((len(outer_indices), len(data["class_order"])), dtype=np.float64),
        "A2_C10_BIAS_GEOMETRIC_GUARD": np.zeros((len(outer_indices), len(data["class_order"])), dtype=np.float64),
    }
    raw_c10_oof = np.zeros_like(anchor_oof["A1_GEOMETRIC_GUARD"])
    guard_oof = np.zeros(len(outer_indices), dtype=bool)
    for inner in outer["inner"]:
        for global_index in inner["valid_indices"]:
            inner_folds[local_by_global[int(global_index)]] = inner["fold"]
        fit_indices = np.asarray(inner["train_indices"], dtype=np.int64)
        valid_indices = np.asarray(inner["valid_indices"], dtype=np.int64)
        local_valid = np.asarray([local_by_global[int(value)] for value in valid_indices], dtype=np.int64)
        anchor = _fit_anchor_boundary(
            data, fit_indices, valid_indices, anchor_cache, source_sha,
            canonical_sha256(inner), f"outer{outer['fold']}-inner{inner['fold']}",
        )
        h1 = anchor["probabilities"]["H1"]
        c10 = anchor["probabilities"]["C10"]
        subinner_probability = np.zeros((len(fit_indices), len(data["class_order"])), dtype=np.float64)
        fit_local = {int(value): index for index, value in enumerate(fit_indices)}
        subcoverage = np.zeros(len(fit_indices), dtype=np.uint8)
        for subinner in inner["subinner"]:
            subfit = np.asarray(subinner["train_indices"], dtype=np.int64)
            subvalid = np.asarray(subinner["valid_indices"], dtype=np.int64)
            probability = _fit_c10_boundary(
                data, subfit, subvalid, anchor_cache, source_sha,
                canonical_sha256(subinner),
                f"outer{outer['fold']}-inner{inner['fold']}-sub{subinner['fold']}-C10",
            )
            positions = np.asarray([fit_local[int(value)] for value in subvalid], dtype=np.int64)
            subinner_probability[positions] = probability
            subcoverage[positions] += 1
        if not np.all(subcoverage == 1):
            raise ValueError("sub-inner C10 OOF coverage is not exactly once")
        a1 = build_anchor_oof(
            h1, c10, anchor["guard_mask"], subinner_probability, data["y_int"][fit_indices],
            branch="A1_GEOMETRIC_GUARD",
        )
        a2 = build_anchor_oof(
            h1, c10, anchor["guard_mask"], subinner_probability, data["y_int"][fit_indices],
            branch="A2_C10_BIAS_GEOMETRIC_GUARD",
        )
        anchor_oof["A1_GEOMETRIC_GUARD"][local_valid] = a1["probability"]
        anchor_oof["A2_C10_BIAS_GEOMETRIC_GUARD"][local_valid] = a2["probability"]
        raw_c10_oof[local_valid] = c10
        guard_oof[local_valid] = anchor["guard_mask"]

    def fit_predict(declaration, train_local, valid_local):
        global_train = outer_indices[np.asarray(train_local)]
        global_valid = outer_indices[np.asarray(valid_local)]
        return _fit_candidate_boundary(
            declaration, data, global_train, global_valid, base_cache, source_sha,
            canonical_sha256([outer["fold"], declaration["candidate_id"], np.asarray(valid_local).tolist()]),
        )

    bank = run_base_bank(
        declarations, inner_folds, data["y_label"][outer_indices], data["class_order"],
        base_cache, fit_predict, input_sha256=TRAIN_SHA256,
        parser_sha256=sha256_file(Path(__file__).parent / "vendor/mcmp_repr.py"),
        source_sha256=source_sha, ids=data["ids"][outer_indices], groups=data["groups"][outer_indices],
        partition_sha256=canonical_sha256(outer),
    )
    by_slot = {slot: {} for slot in ("project", "kim", "ahn", "cross")}
    declaration_by_id = {row["candidate_id"]: row for row in declarations}
    for candidate_id, probability in bank["probabilities"].items():
        by_slot[slot_for(declaration_by_id[candidate_id])][candidate_id] = probability
    return outer_indices, anchor_oof, raw_c10_oof, guard_oof, by_slot, declaration_by_id


def _run_outer_fold(outer, data, declarations, run_root, source_sha):
    fold_root = Path(run_root) / "outer" / f"fold_{outer['fold']}"
    complete_path = fold_root / "fold_probability.npz"
    if complete_path.is_file():
        with np.load(complete_path, allow_pickle=False) as stored:
            return {"fold": outer["fold"], "valid_indices": stored["valid_indices"], "probability": stored["probability"]}
    outer_indices, anchors, inner_c10, guard, bank, declaration_by_id = _inner_work(
        outer, data, declarations, run_root, source_sha,
    )
    selection = select_inner_recipe(anchors, bank, data["y_int"][outer_indices], guard)
    _save_joblib_once(fold_root / "selection.joblib", selection)
    outer_valid = np.asarray(outer["valid_indices"], dtype=np.int64)
    outer_cache = ImmutableProbabilityCache(Path(run_root) / "cache" / f"outer_{outer['fold']}" / "refit")

    def refit_predict(plan, train_indices, valid_indices):
        base = {}
        for candidate_id in plan["base_candidate_ids"]:
            probability, _ = _fit_candidate_boundary(
                declaration_by_id[candidate_id], data, outer_indices, outer_valid,
                outer_cache, source_sha, canonical_sha256([outer["fold"], "refit", candidate_id]),
            )
            base[candidate_id] = probability
        anchor = _fit_anchor_boundary(
            data, outer_indices, outer_valid, outer_cache, source_sha,
            canonical_sha256([outer["fold"], "anchor_refit"]), f"outer{outer['fold']}-refit-anchor",
        )
        h1 = anchor["probabilities"]["H1"]
        c10 = anchor["probabilities"]["C10"]
        output_anchors = {}
        if "A1_GEOMETRIC_GUARD" in plan["anchor_branches"]:
            output_anchors["A1_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1, c10), h1, anchor["guard_mask"],
            )
        if "A2_C10_BIAS_GEOMETRIC_GUARD" in plan["anchor_branches"]:
            bias, _ = fit_c10_log_bias(inner_c10, data["y_int"][outer_indices])
            output_anchors["A2_C10_BIAS_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1, apply_log_bias(c10, bias)), h1, anchor["guard_mask"],
            )
        return {"anchors": output_anchors, "base": base, "guard_mask": anchor["guard_mask"]}

    predicted = select_then_predict_outer(selection, outer_indices, outer_valid, refit_predict)
    fold_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        complete_path, valid_indices=outer_valid, probability=predicted["probability"],
        selected_recipe_id=np.asarray(predicted["selected_recipe_id"]),
    )
    _write_json_once(fold_root / "COMPLETE.json", {
        **predicted["receipt"], "fold": outer["fold"],
        "fold_probability_sha256": sha256_file(complete_path),
    })
    return {"fold": outer["fold"], "valid_indices": outer_valid, "probability": predicted["probability"]}


def _preflight(config):
    closure = read_json(config["source_closure_json"])
    verify_source_closure(Path(__file__).parent, closure)
    if closure["composite_sha256"] != config["source_composite_sha256"]:
        raise ValueError("runtime source closure hash mismatch")
    data = _load_training(config)
    admission = audit_admission(Path(__file__).parent)
    if admission["admission_sha256"] != config["admission_sha256"]:
        raise ValueError("runtime admission hash mismatch")
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    if canonical_sha256(partitions) != config["partition_sha256"]:
        raise ValueError("runtime partition hash mismatch")
    return closure, data, admission, partitions


def _resource_snapshot(config):
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    available_kib = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    if available_kib is None:
        raise ValueError("MemAvailable is unavailable")
    host_available = available_kib * 1024
    disk_probe = Path(config["run_root"]).parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    disk_available = shutil.disk_usage(disk_probe).free
    gpu_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if len(gpu_text) != 1:
        raise ValueError("exactly one visible GPU is required")
    gpu_free = int(gpu_text[0].strip())
    gates = {
        "host_available_bytes": host_available,
        "disk_available_bytes": disk_available,
        "gpu_free_mib": gpu_free,
    }
    required = {
        "host_available_bytes": int(config["minimum_host_available_bytes"]),
        "disk_available_bytes": int(config["minimum_disk_free_bytes"]),
        "gpu_free_mib": int(config["minimum_gpu_free_mib"]),
    }
    for name, minimum in required.items():
        if gates[name] < minimum:
            raise ValueError(f"resource gate failed: {name}")
    return {**gates, "minimums": required, "gpu_device": "cuda:0", "cpu_threads": 4}


def preflight_workflow(config):
    closure, data, admission, partitions = _preflight(config)
    resources = _resource_snapshot(config)
    from runner import status
    worker = status(config["run_root"])
    if worker.get("runner_live"):
        raise ValueError("an owned worker is already running")
    return {
        "schema_version": "TEAM_REAL_PREFLIGHT_V1",
        "status": "PASS",
        "rows": len(data["ids"]),
        "classes": len(data["class_order"]),
        "declared": admission["declared_count"],
        "executable": admission["counts"]["EXECUTABLE"],
        "blocked": admission["counts"]["BLOCKED"],
        "outer_folds": len(partitions["outer"]),
        "source_composite_sha256": closure["composite_sha256"],
        "resources": resources,
        "owned_worker_live": False,
        "run_root_created_by_preflight": False,
        "fit_calls": 0,
        "test_read": False,
        "server_score_used": False,
        "external_data_used": False,
    }


def run_workflow(config, run_root):
    """Execute the approved real nested bundle once; safe to resume."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    closure, data, admission, partitions = _preflight(config)
    _resource_snapshot(config)
    from runner import freeze_runtime
    freeze_runtime(run_root, {
        "source_sha256": closure["composite_sha256"],
        "runtime_sha256": config["_runtime_config_sha256"],
        "data_sha256": config["train_sha256"],
        "partition_sha256": config["partition_sha256"],
        "admission_sha256": config["admission_sha256"],
        "class_order_sha256": canonical_sha256(data["class_order"]),
    })
    _write_json_once(run_root / "ADMISSION.json", admission)
    _write_json_once(run_root / "PARTITIONS.json", partitions)
    declarations = [row for row in admission["declarations"] if row["status"] == "EXECUTABLE"]
    state_path = run_root / "STATE.json"
    state = read_json(state_path) if state_path.is_file() else {
        "schema_version": "TEAM_REAL_WORKFLOW_STATE_V1", "status": "RUNNING", "outer_complete": [],
        "candidate": None, "outer_index": None, "inner_index": None, "subinner_index": None,
        "completion_counts": {"outer_folds": 0, "total_outer_folds": 5}, "test_read": False,
    }
    results = []
    for outer in partitions["outer"]:
        state["outer_index"] = outer["fold"]
        atomic_write_json(state_path, state)
        result = _run_outer_fold(outer, data, declarations, run_root, closure["composite_sha256"])
        results.append(result)
        if outer["fold"] not in state["outer_complete"]:
            state["outer_complete"].append(outer["fold"])
        state["completion_counts"]["outer_folds"] = len(state["outer_complete"])
        atomic_write_json(state_path, state)
        append_jsonl(run_root / "events.jsonl", {"event": "OUTER_FOLD_COMPLETE", "fold": outer["fold"]})
    final_path = run_root / "FINAL_PROCEDURE_OOF.npz"
    if not final_path.exists():
        receipt = assemble_final_oof(
            results, data["ids"], data["y_int"], data["groups"], data["folds"], data["class_order"], final_path,
        )
        _write_json_once(run_root / "FINAL_PROCEDURE_OOF_RECEIPT.json", receipt)
    receipt = read_json(run_root / "FINAL_PROCEDURE_OOF_RECEIPT.json")
    score = score_final_oof(load_final_oof(final_path, receipt))
    milestone = milestone_status(score["macro_f1"])
    state.update(status="COMPLETE", outer_index=None, final_score=score, milestone=milestone)
    atomic_write_json(state_path, state)
    atomic_write_json(run_root / "COMPLETE.json", {
        "schema_version": "TEAM_REAL_WORKFLOW_COMPLETE_V1", "status": "COMPLETE",
        "score": score, "milestone": milestone, "automatic_next_bundle": False, "submission": False,
    })
    return state


def _synthetic_probability(seed, rows, classes):
    rng = np.random.default_rng(seed)
    value = rng.random((rows, classes))
    return value / value.sum(axis=1, keepdims=True)


def run_synthetic_acceptance(run_root, source_manifest, training_fixture, *, fail_after_phase=None):
    """Reduced end-to-end path used only by local/Windows acceptance tests."""
    from runner import PHASES, run_resumable
    run_root = Path(run_root)
    artifacts = run_root / "synthetic"

    def feature_state():
        content = Path(training_fixture).read_text(encoding="utf-8")
        artifacts.mkdir(parents=True, exist_ok=True)
        _write_json_once(artifacts / "feature.json", {"training_fixture_sha256": hashlib.sha256(content.encode()).hexdigest(), "test_read": False})
        return {"training_fixture_only": True, "test_read": False}

    def model():
        admission = audit_admission(Path(__file__).parent)
        _write_json_once(artifacts / "admission.json", admission)
        return {"declared": 128, "executable": 124, "blocked": 4, "real_fit_calls": 0}

    def inner_oof():
        y = np.asarray([index % 3 for index in range(60)], dtype=np.int64)
        arrays = {"y": y, "guard": np.zeros(60, dtype=bool)}
        for name, seed in (("A1", 1), ("A2", 2), ("P", 3), ("K", 4), ("H", 5), ("X", 6)):
            arrays[name] = _synthetic_probability(seed, 60, 3)
        np.savez_compressed(artifacts / "inner_oof.npz", **arrays)
        return {"rows": 60, "classes": 3, "coverage": 1, "test_read": False}

    def round_manifest():
        with np.load(artifacts / "inner_oof.npz", allow_pickle=False) as stored:
            anchors = {"A1": stored["A1"], "A2": stored["A2"]}
            bank = {"project": {"P": stored["P"]}, "kim": {"K": stored["K"]},
                    "ahn": {"H": stored["H"]}, "cross": {"X": stored["X"]}}
            selection = select_inner_recipe(anchors, bank, stored["y"], stored["guard"])
        _save_joblib_once(artifacts / "selection.joblib", selection)
        sample = list(selection["round0_by_id"].values())[:2]
        meta, meta_receipt = run_meta_round(sample, np.asarray([index % 3 for index in range(60)]), round_index=1)
        return {"round0": 280, "meta_round_evaluated": len(meta), "meta_complete": meta_receipt["complete"], "test_read": False}

    def outer_refit():
        ids = np.asarray([f"SYN{i:03d}" for i in range(100)])
        folds = np.asarray([index % 5 for index in range(100)], dtype=np.int16)
        results = []
        for fold in range(5):
            indices = np.flatnonzero(folds == fold)
            results.append({"fold": fold, "valid_indices": indices,
                            "probability": _synthetic_probability(100 + fold, len(indices), 3)})
        _save_joblib_once(artifacts / "outer_results.joblib", results)
        return {"outer_folds": 5, "outer_labels_accepted": False, "test_read": False}

    def fold_completion():
        ids = np.asarray([f"SYN{i:03d}" for i in range(100)])
        y = np.asarray([index % 3 for index in range(100)], dtype=np.int64)
        groups = np.asarray([f"G{index:03d}" for index in range(100)])
        folds = np.asarray([index % 5 for index in range(100)], dtype=np.int16)
        path = artifacts / "FINAL_PROCEDURE_OOF.npz"
        if not path.exists():
            receipt = assemble_final_oof(joblib.load(artifacts / "outer_results.joblib"), ids, y, groups, folds, ["C0", "C1", "C2"], path)
            _write_json_once(artifacts / "FINAL_RECEIPT.json", receipt)
        score = score_final_oof(load_final_oof(path, read_json(artifacts / "FINAL_RECEIPT.json")))
        return {"exact_outer_coverage": True, "macro_f1": score["macro_f1"], "test_read": False}

    callbacks = {
        "feature_state": feature_state,
        "model": model,
        "inner_oof": inner_oof,
        "round_manifest": round_manifest,
        "outer_refit": outer_refit,
        "fold_completion": fold_completion,
    }
    return run_resumable(run_root, callbacks, source_manifest, fail_after_phase=fail_after_phase)
