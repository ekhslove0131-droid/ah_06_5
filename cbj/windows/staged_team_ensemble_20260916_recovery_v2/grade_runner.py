"""Graded recovery runner: A0 -> A1 -> B1 -> B2, with a CSV gate per grade.

Selection is made only from outer-0 inner OOF.  Once frozen, the selected
recipe is replayed across the five outer folds and the matching test
predictions are averaged.  Full OOF is diagnostic evidence and is never fed
back into selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from anchor_pipeline import build_anchor_oof, prepare_anchor_features, prepare_c10_features
from atomic_state import append_jsonl, atomic_write_json, read_json
from base_bank import _lookup_key, run_base_bank
from base_receipts import canonical_sha256, probability_sha256, sequence_sha256
from cache_store import ImmutableProbabilityCache, make_cache_key
from contracts import CLASS_ORDER, TRAIN_SHA256
from eligibility_audit import audit_admission, slot_for
from ensemble_grid import run_round0
from feature_transforms import FeatureContext
from log_bias import apply_log_bias, fit_c10_log_bias
from meta_rounds import run_meta_round
from model_adapters import effective_recipe, fit_model, predict_checked
from numeric_views import build_numeric41
from outer_procedure import apply_selected_recipe, build_refit_plan
from partition_contract import build_partitions
from promotion import disagreement
from runner import _runner_live, launch_detached
from source_closure import sha256_file, verify_source_closure
from support_guard import geometric_pool, guarded_probability
from workflow import (
    _feature_order_hash, _fit_candidate_boundary, _load_training,
    _prepare_candidate, _resource_snapshot,
)


OLD_SOURCE_SHA256 = "6192a97496fbb7bb17fe52c259d1f9574a01c6884c9649e1a52fa73904fa9c43"
OLD_PARSER_SHA256 = "0673f10734aff57d6b356ee21661a45a9301b037e712a4a7f2d6d26fcff0b30b"
HISTORICAL_COMPARATOR = 0.5204059148667912

GRADE_ADDITIONS = {
    "A0": (),
    "A1": (
        "TEAM-T4-0003", "TEAM-T4-0005", "TEAM-T5-0003",
        "TEAM-T5-0007", "TEAM-T6-0007", "TEAM-T6-0008",
    ),
    "B1": (
        "TEAM-T4-0007", "TEAM-T5-0001", "TEAM-T5-0004",
        "TEAM-T5-0005", "TEAM-T6-0003", "TEAM-T6-0005", "TEAM-T6-0006",
    ),
    "B2": (
        "TEAM-T5-0002", "TEAM-T5-0006", "TEAM-T5-0008",
        "TEAM-T6-0001", "TEAM-T6-0002", "TEAM-T6-0004",
    ),
}
GRADE_ORDER = tuple(GRADE_ADDITIONS)


def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _source_identity(root):
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
    rows = [{"path": name, "sha256": sha256_file(root / name)} for name in names]
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest(), rows


def _declarations(root):
    report = audit_admission(root)
    executable = [row for row in report["declarations"] if row["status"] == "EXECUTABLE"]
    by_id = {row["candidate_id"]: row for row in executable}
    additions = {candidate for values in GRADE_ADDITIONS.values() for candidate in values}
    missing = additions - set(by_id)
    if missing:
        raise ValueError(f"graded candidate IDs are unavailable: {sorted(missing)}")
    a0 = [row for row in executable if row["candidate_id"] not in additions]
    if len(a0) != 105:
        raise ValueError(f"A0 must contain exactly 105 completed candidates, got {len(a0)}")
    return report, a0, by_id


def _old_cache(config, kind):
    return ImmutableProbabilityCache(Path(config["old_run_root"]) / "cache" / "outer_0" / kind)


def _verify_probability(probability, rows, classes=26):
    value = np.asarray(probability, dtype=np.float64)
    if value.shape != (rows, classes):
        raise ValueError("probability shape mismatch")
    if not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(1), 1, atol=1e-8):
        raise ValueError("probability values are invalid")
    return value


def _load_old_base_oof(config, data, outer, declarations):
    cache = _old_cache(config, "base")
    outer_indices = np.asarray(outer["train_indices"], dtype=np.int64)
    local_by_global = {int(value): index for index, value in enumerate(outer_indices)}
    partition_sha = canonical_sha256(outer)
    output = {row["candidate_id"]: np.zeros((len(outer_indices), 26), dtype=np.float64) for row in declarations}
    coverage = {row["candidate_id"]: np.zeros(len(outer_indices), dtype=np.uint8) for row in declarations}
    for declaration in declarations:
        for inner in outer["inner"]:
            valid_global = np.asarray(inner["valid_indices"], dtype=np.int64)
            train_global = np.asarray(inner["train_indices"], dtype=np.int64)
            valid_local = np.asarray([local_by_global[int(value)] for value in valid_global], dtype=np.int64)
            train_local = np.asarray([local_by_global[int(value)] for value in train_global], dtype=np.int64)
            prebindings = {
                "schema_version": "BASE_BANK_LOOKUP_V1",
                "candidate_id": declaration["candidate_id"],
                "declaration_sha256": declaration["config_hash"],
                "fold": int(inner["fold"]),
                "input_sha256": TRAIN_SHA256,
                "parser_sha256": OLD_PARSER_SHA256,
                "source_sha256": OLD_SOURCE_SHA256,
                "fit_ids_sha256": sequence_sha256(data["ids"][outer_indices][train_local]),
                "fit_y_sha256": sequence_sha256(data["y_label"][outer_indices][train_local]),
                "fit_groups_sha256": sequence_sha256(data["groups"][outer_indices][train_local]),
                "partition_sha256": canonical_sha256([partition_sha, int(inner["fold"]), valid_local.tolist()]),
                "parameters_sha256": declaration["config_hash"],
            }
            key = cache.resolve_index(_lookup_key(prebindings))
            if key is None:
                raise ValueError(f"old cache index is missing: {declaration['candidate_id']} fold {inner['fold']}")
            receipt = read_json(cache.artifact_path(key) / "COMPLETE.json")
            expected_bindings = {name: value for name, value in prebindings.items() if name.endswith("sha256")}
            actual_bindings = receipt.get("bindings", {})
            for name, value in expected_bindings.items():
                if actual_bindings.get(name) != value:
                    raise ValueError(f"old cache index lineage mismatch: {declaration['candidate_id']} {name}")
            if key != make_cache_key(actual_bindings, supervised=True):
                raise ValueError("old cache key does not match its receipt bindings")
            probability, _ = cache._verify(key, receipt["bindings"])
            probability = _verify_probability(probability, len(valid_local))
            output[declaration["candidate_id"]][valid_local] = probability
            coverage[declaration["candidate_id"]][valid_local] += 1
    if any(not np.all(value == 1) for value in coverage.values()):
        raise ValueError("old A0 OOF coverage is incomplete")
    return output


def _anchor_artifact_map(config):
    cache = _old_cache(config, "anchors")
    rows = {}
    for complete in cache.artifacts.glob("*/COMPLETE.json"):
        receipt = read_json(complete)
        bindings = receipt["bindings"]
        if bindings.get("source_sha256") != OLD_SOURCE_SHA256:
            raise ValueError("old anchor source identity changed")
        if bindings.get("input_sha256") != TRAIN_SHA256 or bindings.get("parser_sha256") != OLD_PARSER_SHA256:
            raise ValueError("old anchor input or parser identity changed")
        if receipt["key"] != make_cache_key(bindings, supervised=True):
            raise ValueError("old anchor cache key does not match receipt bindings")
        key = (
            bindings["declaration_sha256"], bindings["fit_ids_sha256"],
            bindings["fit_y_sha256"], bindings["fit_groups_sha256"],
            bindings["partition_sha256"], bindings["parameters_sha256"],
        )
        if key in rows:
            raise ValueError("old anchor cache contains an ambiguous fit identity")
        probability, _ = cache._verify(receipt["key"], bindings)
        rows[key] = probability
    if len(rows) != 15:
        raise ValueError(f"old anchor cache must contain exactly 15 fits, got {len(rows)}")
    return rows


def _old_anchor_probability(anchor_map, recipe, data, indices, namespace, partition_sha):
    indices = np.asarray(indices, dtype=np.int64)
    key = (
        recipe["config_hash"], sequence_sha256(data["ids"][indices]),
        sequence_sha256(data["y_label"][indices]), sequence_sha256(data["groups"][indices]),
        canonical_sha256([namespace, partition_sha]), canonical_sha256(recipe),
    )
    if key not in anchor_map:
        raise ValueError(f"old anchor probability missing for {recipe['candidate_id']}")
    return anchor_map[key]


def _context(data, fit_indices, valid_indices):
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    train = data["frame"].iloc[fit_indices][data["genes"]].reset_index(drop=True)
    valid = data["frame"].iloc[valid_indices][data["genes"]].reset_index(drop=True)
    numeric = build_numeric41(train, valid, data["genes"], data["groups"][fit_indices])
    context = FeatureContext(
        gene_cols=data["genes"], train_ids=data["ids"][fit_indices], valid_ids=data["ids"][valid_indices],
        train_groups=data["groups"][fit_indices], train_y=data["y_label"][fit_indices],
        class_order=data["class_order"], numeric41_train=numeric["train"], numeric41_valid=numeric["valid"],
    )
    return train, valid, context


def _load_old_inner_evidence(config, data, outer, a0_declarations):
    from anchor_pipeline import anchor_model_recipes
    from support_guard import support_guard_mask

    outer_indices = np.asarray(outer["train_indices"], dtype=np.int64)
    local_by_global = {int(value): index for index, value in enumerate(outer_indices)}
    inner_folds = np.empty(len(outer_indices), dtype=np.int16)
    anchors = {name: np.zeros((len(outer_indices), 26), dtype=np.float64) for name in (
        "A1_GEOMETRIC_GUARD", "A2_C10_BIAS_GEOMETRIC_GUARD",
    )}
    raw_c10 = np.zeros((len(outer_indices), 26), dtype=np.float64)
    guard = np.zeros(len(outer_indices), dtype=bool)
    recipes = anchor_model_recipes()
    anchor_map = _anchor_artifact_map(config)
    for inner in outer["inner"]:
        fit_indices = np.asarray(inner["train_indices"], dtype=np.int64)
        valid_indices = np.asarray(inner["valid_indices"], dtype=np.int64)
        local_valid = np.asarray([local_by_global[int(value)] for value in valid_indices], dtype=np.int64)
        inner_folds[local_valid] = int(inner["fold"])
        namespace = f"outer0-inner{inner['fold']}"
        inner_partition = canonical_sha256(inner)
        h1 = _old_anchor_probability(
            anchor_map, recipes["H1"], data, fit_indices, f"{namespace}-H1",
            canonical_sha256([inner_partition, "H1"]),
        )
        c10 = _old_anchor_probability(
            anchor_map, recipes["C10"], data, fit_indices, f"{namespace}-C10",
            canonical_sha256([inner_partition, "C10"]),
        )
        if h1.shape[0] != len(valid_indices) or c10.shape[0] != len(valid_indices):
            raise ValueError("old anchor rows differ from the frozen inner partition")
        train, valid, context = _context(data, fit_indices, valid_indices)
        _, symbolic_train, symbolic_valid = prepare_c10_features(train, valid, context)
        inner_guard, _ = support_guard_mask(symbolic_train, symbolic_valid)
        sub_oof = np.zeros((len(fit_indices), 26), dtype=np.float64)
        fit_local = {int(value): index for index, value in enumerate(fit_indices)}
        subcoverage = np.zeros(len(fit_indices), dtype=np.uint8)
        for subinner in inner["subinner"]:
            subfit = np.asarray(subinner["train_indices"], dtype=np.int64)
            subvalid = np.asarray(subinner["valid_indices"], dtype=np.int64)
            subprob = _old_anchor_probability(
                anchor_map, recipes["C10"], data, subfit,
                f"outer0-inner{inner['fold']}-sub{subinner['fold']}-C10",
                canonical_sha256(subinner),
            )
            positions = np.asarray([fit_local[int(value)] for value in subvalid], dtype=np.int64)
            sub_oof[positions] = _verify_probability(subprob, len(positions))
            subcoverage[positions] += 1
        if not np.all(subcoverage == 1):
            raise ValueError("old sub-inner anchor coverage is incomplete")
        for name in anchors:
            anchors[name][local_valid] = build_anchor_oof(
                h1, c10, inner_guard, sub_oof, data["y_int"][fit_indices], branch=name,
            )["probability"]
        raw_c10[local_valid] = c10
        guard[local_valid] = inner_guard
    base = _load_old_base_oof(config, data, outer, a0_declarations)
    return outer_indices, inner_folds, anchors, raw_c10, guard, base


def _new_stage_probabilities(config, run_root, data, outer, inner_folds, declarations, source_sha):
    if not declarations:
        return {}, []
    outer_indices = np.asarray(outer["train_indices"], dtype=np.int64)
    cache = ImmutableProbabilityCache(Path(run_root) / "cache" / "outer_0" / "graded_base")
    parser_sha = sha256_file(Path(__file__).parent / "vendor/mcmp_repr.py")
    output = {}
    failures = []
    for declaration in declarations:
        job_path = Path(run_root) / "jobs" / f"{declaration['candidate_id']}.json"
        atomic_write_json(job_path, {
            "candidate_id": declaration["candidate_id"], "status": "RUNNING",
            "config_hash": declaration["config_hash"],
        })
        def fit_predict(current, train_local, valid_local):
            started = time.time()
            append_jsonl(Path(run_root) / "events.jsonl", {
                "event": "INNER_COMPONENT_FIT_START", "candidate_id": current["candidate_id"],
                "inner_fold": int(inner_folds[np.asarray(valid_local)[0]]), "time_unix": started,
            })
            train_global = outer_indices[np.asarray(train_local)]
            valid_global = outer_indices[np.asarray(valid_local)]
            result = _fit_candidate_boundary(
                current, data, train_global, valid_global, cache, source_sha,
                canonical_sha256([0, current["candidate_id"], np.asarray(valid_local).tolist()]),
            )
            append_jsonl(Path(run_root) / "events.jsonl", {
                "event": "INNER_COMPONENT_FIT_COMPLETE", "candidate_id": current["candidate_id"],
                "inner_fold": int(inner_folds[np.asarray(valid_local)[0]]),
                "elapsed_seconds": time.time() - started, "time_unix": time.time(),
            })
            return result
        try:
            result = run_base_bank(
                [declaration], inner_folds, data["y_label"][outer_indices], data["class_order"],
                cache, fit_predict, input_sha256=TRAIN_SHA256, parser_sha256=parser_sha,
                source_sha256=source_sha, ids=data["ids"][outer_indices], groups=data["groups"][outer_indices],
                partition_sha256=canonical_sha256(outer),
            )
        except Exception:
            atomic_write_json(job_path, {
                "candidate_id": declaration["candidate_id"], "status": "FAILED",
                "config_hash": declaration["config_hash"], "traceback": traceback.format_exc(),
            })
            append_jsonl(Path(run_root) / "events.jsonl", {
                "event": "CANDIDATE_FAILED", "candidate_id": declaration["candidate_id"],
            })
            failures.append(declaration["candidate_id"])
            continue
        probability = result["probabilities"][declaration["candidate_id"]]
        output[declaration["candidate_id"]] = probability
        atomic_write_json(job_path, {
            "candidate_id": declaration["candidate_id"], "status": "COMPLETE",
            "config_hash": declaration["config_hash"], "reused_folds": result["reused_folds"],
            "probability_sha256": probability_sha256(probability),
        })
    return output, failures


def _bounded_promote(candidates):
    ordered = sorted((row for row in candidates if row.macro_f1 >= 0.55), key=lambda row: (-row.macro_f1, row.recipe_id))
    retained = []
    for candidate in ordered:
        if all(disagreement(candidate, previous) >= 0.03 for previous in retained):
            retained.append(candidate)
            if len(retained) == 4:
                break
    return retained


def _select(anchors, bank_by_slot, y, guard):
    round0 = run_round0(anchors, bank_by_slot, y, guard)
    best = sorted(round0, key=lambda row: (-row.macro_f1, row.recipe_id))[0]
    graph = {row.recipe_id: row for row in round0}
    retained = _bounded_promote(round0)
    manifests = [{"round": 0, "evaluated": 280, "retained": len(retained), "best_macro_f1": best.macro_f1}]
    for round_index in range(1, 3):
        if len(retained) < 2:
            break
        rows, manifest = run_meta_round(retained, y, round_index=round_index)
        graph.update({row.recipe_id: row for row in rows})
        round_best = sorted(rows, key=lambda row: (-row.macro_f1, row.recipe_id))[0]
        improved = round_best.macro_f1 > best.macro_f1 + 1e-12
        if improved:
            best = round_best
        retained = _bounded_promote(rows)
        manifests.append({**manifest, "retained": len(retained), "strict_improvement": improved})
        if not improved:
            break
    return {"selected": best, "candidate_graph": graph, "manifests": manifests}


def _slim_selection(selection):
    selected = selection["selected"]
    graph = selection["candidate_graph"]
    keep = {}
    def visit(candidate):
        if candidate.recipe_id in keep:
            return
        keep[candidate.recipe_id] = candidate
        if candidate.round_index > 0:
            for weight, recipe_id in zip(candidate.weights, candidate.slot_candidate_ids):
                if weight > 0:
                    visit(graph[recipe_id])
    visit(selected)
    return {"selected": selected, "candidate_graph": keep, "manifests": selection["manifests"]}


def _freeze_selection(path, grade, candidate_ids, selection=None):
    path = Path(path)
    receipt_path = path.with_name("SELECTION.json")
    membership_sha = canonical_sha256(list(candidate_ids))
    if path.is_file() or receipt_path.is_file():
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError("frozen selection files are incomplete")
        receipt = read_json(receipt_path)
        if receipt.get("grade") != grade or receipt.get("candidate_membership_sha256") != membership_sha:
            raise ValueError("frozen selection membership changed")
        if receipt.get("selection_sha256") != sha256_file(path):
            raise ValueError("frozen selection hash mismatch")
        loaded = joblib.load(path)
        if loaded["selected"].recipe_id != receipt.get("selected_recipe_id"):
            raise ValueError("frozen selection recipe mismatch")
        return loaded, receipt
    if selection is None:
        raise ValueError("selection is required for a new freeze")
    selection = _slim_selection(selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    joblib.dump(selection, temporary, compress=3)
    os.replace(temporary, path)
    receipt = {
        "schema_version": "GRADED_SELECTION_FREEZE_V1", "grade": grade,
        "candidate_membership_sha256": membership_sha, "candidate_count": len(candidate_ids),
        "selection_sha256": sha256_file(path),
        "selected_recipe_id": selection["selected"].recipe_id,
        "inner_macro_f1": float(selection["selected"].macro_f1),
        "test_read_before_freeze": False,
    }
    atomic_write_json(receipt_path, receipt)
    return selection, receipt


def _build_bank(declarations, probabilities):
    bank = {slot: {} for slot in ("project", "kim", "ahn", "cross")}
    for declaration in declarations:
        candidate_id = declaration["candidate_id"]
        if candidate_id not in probabilities:
            raise ValueError(f"candidate OOF is missing: {candidate_id}")
        bank[slot_for(declaration)][candidate_id] = probabilities[candidate_id]
    return bank


def _load_inference(config, data):
    test_path = Path(config["test_csv"])
    sample_path = Path(config["sample_submission_csv"])
    if sha256_file(test_path) != config["test_sha256"] or sha256_file(sample_path) != config["sample_submission_sha256"]:
        raise ValueError("inference input hash mismatch")
    test = pd.read_csv(test_path, encoding="utf-8")
    sample = pd.read_csv(sample_path, encoding="utf-8")
    if list(test.columns) != ["ID", *data["genes"]]:
        raise ValueError("test columns differ from frozen training feature order")
    if list(sample.columns) != ["ID", "SUBCLASS"]:
        raise ValueError("sample submission schema mismatch")
    if not np.array_equal(test["ID"].astype(str).to_numpy(), sample["ID"].astype(str).to_numpy()):
        raise ValueError("test and sample submission ID order mismatch")
    if test["ID"].duplicated().any():
        raise ValueError("test ID values are duplicated")
    return test, sample


def _combined_context(data, train_indices, valid_indices, test):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    train = data["frame"].iloc[train_indices][data["genes"]].reset_index(drop=True)
    valid_train = data["frame"].iloc[valid_indices][data["genes"]].reset_index(drop=True)
    valid = pd.concat([valid_train, test[data["genes"]]], axis=0, ignore_index=True)
    valid_ids = np.r_[data["ids"][valid_indices], test["ID"].astype(str).to_numpy()]
    numeric = build_numeric41(train, valid, data["genes"], data["groups"][train_indices])
    context = FeatureContext(
        gene_cols=data["genes"], train_ids=data["ids"][train_indices], valid_ids=valid_ids,
        train_groups=data["groups"][train_indices], train_y=data["y_label"][train_indices],
        class_order=data["class_order"], numeric41_train=numeric["train"], numeric41_valid=numeric["valid"],
    )
    return train, valid, context


def _component_paths(run_root, name, fold):
    safe = name.replace("/", "_")
    root = Path(run_root) / "deployment" / "components" / safe / f"fold_{fold}"
    return root / "probability.npz", root / "COMPLETE.json"


def _component_bindings(name, declaration_hash, fold, data, train_indices, valid_indices, test_ids, source_sha):
    return {
        "component": name, "declaration_sha256": declaration_hash, "fold": int(fold),
        "source_sha256": source_sha,
        "train_ids_sha256": sequence_sha256(data["ids"][train_indices]),
        "train_y_sha256": sequence_sha256(data["y_label"][train_indices]),
        "train_groups_sha256": sequence_sha256(data["groups"][train_indices]),
        "valid_ids_sha256": sequence_sha256(data["ids"][valid_indices]),
        "test_ids_sha256": sequence_sha256(test_ids),
        "class_order": list(data["class_order"]),
    }


def _save_component(run_root, name, fold, bindings, valid_probability, test_probability, *, guard=None):
    npz_path, receipt_path = _component_paths(run_root, name, fold)
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("bindings") != bindings or receipt.get("npz_sha256") != sha256_file(npz_path):
            raise ValueError("component cache lineage mismatch")
        with np.load(npz_path, allow_pickle=False) as stored:
            return stored["valid"], stored["test"], stored["guard"] if "guard" in stored else None
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_name(f".{npz_path.name}.partial")
    payload = {"valid": valid_probability, "test": test_probability}
    if guard is not None:
        payload["guard"] = np.asarray(guard, dtype=bool)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, npz_path)
    atomic_write_json(receipt_path, {
        "schema_version": "GRADED_COMPONENT_CACHE_V1", "bindings": bindings,
        "npz_sha256": sha256_file(npz_path),
        "valid_probability_sha256": probability_sha256(valid_probability),
        "test_probability_sha256": probability_sha256(test_probability),
    })
    return valid_probability, test_probability, guard


def _fit_component(config, run_root, data, test, declaration, fold, train_indices, valid_indices, source_sha):
    name = declaration["candidate_id"]
    bindings = _component_bindings(
        name, declaration["config_hash"], fold, data, train_indices, valid_indices,
        test["ID"].astype(str).to_numpy(), source_sha,
    )
    npz_path, receipt_path = _component_paths(run_root, name, fold)
    if receipt_path.is_file():
        return _save_component(run_root, name, fold, bindings, None, None)[:2]
    started = time.time()
    append_jsonl(Path(run_root) / "events.jsonl", {
        "event": "DEPLOY_COMPONENT_FIT_START", "component": name, "fold": int(fold),
        "time_unix": started,
    })
    train, valid, context = _combined_context(data, train_indices, valid_indices, test)
    features = _prepare_candidate(declaration, train, valid, context)
    recipe = effective_recipe(declaration)
    artifact = fit_model(recipe, features, context.train_y, None, context.class_order)
    prediction_input = features.valid
    if (recipe.get("transformer") or {}).get("kind") == "AE64":
        prediction_input = {"ae_input": features.valid, "numeric41": features.deferred_numeric_valid}
    probability = predict_checked(artifact, prediction_input, context.class_order)
    boundary = len(valid_indices)
    if probability.shape[0] != boundary + len(test):
        raise ValueError("candidate combined valid/test prediction row count mismatch")
    saved = _save_component(
        run_root, name, fold, bindings, probability[:boundary], probability[boundary:],
    )[:2]
    append_jsonl(Path(run_root) / "events.jsonl", {
        "event": "DEPLOY_COMPONENT_FIT_COMPLETE", "component": name, "fold": int(fold),
        "elapsed_seconds": time.time() - started, "time_unix": time.time(),
    })
    return saved


def _fit_anchor_components(config, run_root, data, test, fold, train_indices, valid_indices, source_sha):
    from anchor_pipeline import anchor_model_recipes
    recipes = anchor_model_recipes()
    test_ids = test["ID"].astype(str).to_numpy()
    cached = {}
    missing = []
    for name in ("H1", "C10"):
        _, receipt_path = _component_paths(run_root, f"ANCHOR-{name}", fold)
        bindings = _component_bindings(
            f"ANCHOR-{name}", recipes[name]["config_hash"], fold, data,
            train_indices, valid_indices, test_ids, source_sha,
        )
        if receipt_path.is_file():
            valid, test_probability, guard = _save_component(
                run_root, f"ANCHOR-{name}", fold, bindings, None, None,
            )
            cached[name] = (valid, test_probability, guard)
        else:
            missing.append(name)
    if not missing:
        guard = cached["C10"][2]
        if guard is None:
            raise ValueError("cached C10 anchor guard is missing")
        return {
            "H1": cached["H1"][:2], "C10": cached["C10"][:2],
            "guard": guard,
        }
    train, valid, context = _combined_context(data, train_indices, valid_indices, test)
    features = prepare_anchor_features(train, valid, context)
    boundary = len(valid_indices)
    output = {name: cached[name][:2] for name in cached}
    for name in missing:
        started = time.time()
        append_jsonl(Path(run_root) / "events.jsonl", {
            "event": "DEPLOY_COMPONENT_FIT_START", "component": f"ANCHOR-{name}",
            "fold": int(fold), "time_unix": started,
        })
        bindings = _component_bindings(
            f"ANCHOR-{name}", recipes[name]["config_hash"], fold, data,
            train_indices, valid_indices, test_ids, source_sha,
        )
        artifact = fit_model(recipes[name], features[name], context.train_y, None, context.class_order)
        probability = predict_checked(artifact, features[name].valid, context.class_order)
        if probability.shape[0] != boundary + len(test):
            raise ValueError("anchor combined valid/test prediction row count mismatch")
        guard_payload = features["guard_mask"] if name == "C10" else None
        valid_probability, test_probability, _ = _save_component(
            run_root, f"ANCHOR-{name}", fold, bindings,
            probability[:boundary], probability[boundary:], guard=guard_payload,
        )
        output[name] = (valid_probability, test_probability)
        append_jsonl(Path(run_root) / "events.jsonl", {
            "event": "DEPLOY_COMPONENT_FIT_COMPLETE", "component": f"ANCHOR-{name}",
            "fold": int(fold), "elapsed_seconds": time.time() - started,
            "time_unix": time.time(),
        })
    guard = cached.get("C10", (None, None, None))[2]
    if guard is None:
        guard = features["guard_mask"]
    return {**output, "guard": guard}


def _deploy_stage(config, run_root, data, partitions, test, selection, inner_c10, outer0_indices, source_sha, declaration_by_id):
    selected = selection["selected"]
    graph = selection["candidate_graph"]
    plan = build_refit_plan(selected, graph)
    bias, bias_receipt = fit_c10_log_bias(inner_c10, data["y_int"][outer0_indices])
    oof = np.zeros((len(data["ids"]), 26), dtype=np.float64)
    coverage = np.zeros(len(data["ids"]), dtype=np.uint8)
    test_sum = np.zeros((len(test), 26), dtype=np.float64)
    for outer in partitions["outer"]:
        fold = int(outer["fold"])
        train_indices = np.asarray(outer["train_indices"], dtype=np.int64)
        valid_indices = np.asarray(outer["valid_indices"], dtype=np.int64)
        anchors_raw = _fit_anchor_components(
            config, run_root, data, test, fold, train_indices, valid_indices, source_sha,
        )
        guard_all = np.asarray(anchors_raw["guard"], dtype=bool)
        boundary = len(valid_indices)
        anchor_valid = {}; anchor_test = {}
        h1_valid, h1_test = anchors_raw["H1"]
        c10_valid, c10_test = anchors_raw["C10"]
        if "A1_GEOMETRIC_GUARD" in plan["anchor_branches"]:
            anchor_valid["A1_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_valid, c10_valid), h1_valid, guard_all[:boundary],
            )
            anchor_test["A1_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_test, c10_test), h1_test, guard_all[boundary:],
            )
        if "A2_C10_BIAS_GEOMETRIC_GUARD" in plan["anchor_branches"]:
            anchor_valid["A2_C10_BIAS_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_valid, apply_log_bias(c10_valid, bias)), h1_valid, guard_all[:boundary],
            )
            anchor_test["A2_C10_BIAS_GEOMETRIC_GUARD"] = guarded_probability(
                geometric_pool(h1_test, apply_log_bias(c10_test, bias)), h1_test, guard_all[boundary:],
            )
        base_valid = {}; base_test = {}
        for candidate_id in plan["base_candidate_ids"]:
            valid_probability, test_probability = _fit_component(
                config, run_root, data, test, declaration_by_id[candidate_id], fold,
                train_indices, valid_indices, source_sha,
            )
            base_valid[candidate_id] = valid_probability
            base_test[candidate_id] = test_probability
        fold_valid = apply_selected_recipe(selected, graph, anchor_valid, base_valid, guard_all[:boundary])
        fold_test = apply_selected_recipe(selected, graph, anchor_test, base_test, guard_all[boundary:])
        oof[valid_indices] = fold_valid
        coverage[valid_indices] += 1
        test_sum += fold_test
    if not np.all(coverage == 1):
        raise ValueError("deployment OOF coverage is incomplete")
    test_probability = test_sum / 5.0
    return oof, test_probability, {"refit_plan": plan, "bias_receipt": bias_receipt}


def _write_stage_predictions(run_root, grade, data, test, oof, test_probability, selection_sha):
    path = Path(run_root) / "grades" / grade / "predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        receipt = read_json(path.with_name("PREDICTIONS.json"))
        if receipt.get("npz_sha256") != sha256_file(path) or receipt.get("selection_sha256") != selection_sha:
            raise ValueError("existing stage prediction lineage mismatch")
        with np.load(path, allow_pickle=False) as stored:
            if (not np.array_equal(stored["ids"], data["ids"])
                    or not np.array_equal(stored["test_ids"], test["ID"].astype(str).to_numpy())
                    or probability_sha256(stored["oof_probability"]) != probability_sha256(oof)
                    or probability_sha256(stored["test_probability"]) != probability_sha256(test_probability)):
                raise ValueError("existing stage predictions differ on resume")
        return path, receipt
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, ids=data["ids"], y=data["y_int"], groups=data["groups"], folds=data["folds"],
            class_order=np.asarray(data["class_order"]), oof_probability=oof,
            test_ids=test["ID"].astype(str).to_numpy(), test_probability=test_probability,
        )
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    receipt = {
        "schema_version": "GRADED_STAGE_PREDICTIONS_V1", "grade": grade,
        "npz_sha256": sha256_file(path), "selection_sha256": selection_sha,
        "ids_sha256": sequence_sha256(data["ids"]), "y_sha256": sequence_sha256(data["y_int"]),
        "groups_sha256": sequence_sha256(data["groups"]), "folds_sha256": sequence_sha256(data["folds"]),
        "test_ids_sha256": sequence_sha256(test["ID"].astype(str).to_numpy()),
        "oof_probability_sha256": probability_sha256(oof),
        "test_probability_sha256": probability_sha256(test_probability),
    }
    atomic_write_json(path.with_name("PREDICTIONS.json"), receipt)
    return path, receipt


def _verify_stage_complete(run_root, grade):
    root = Path(run_root) / "grades" / grade
    complete = read_json(root / "COMPLETE.json")
    submission = Path(complete["submission_csv"])
    selection = root / "selection.joblib"
    predictions = root / "predictions.npz"
    prediction_receipt = read_json(root / "PREDICTIONS.json")
    if complete.get("submission_sha256") != sha256_file(submission):
        raise ValueError("completed grade submission hash mismatch")
    if complete.get("selection_sha256") != sha256_file(selection):
        raise ValueError("completed grade selection hash mismatch")
    if prediction_receipt.get("npz_sha256") != sha256_file(predictions):
        raise ValueError("completed grade prediction hash mismatch")
    if complete.get("predictions_sha256") != prediction_receipt.get("npz_sha256"):
        raise ValueError("completed grade prediction receipt differs")
    frame = pd.read_csv(submission, encoding="utf-8")
    if sequence_sha256(frame["ID"].astype(str).to_numpy()) != complete.get("id_sha256"):
        raise ValueError("completed grade submission IDs changed")
    return complete


def _write_submission(run_root, grade, sample, test_probability, class_order, receipt):
    if test_probability.shape != (len(sample), len(class_order)):
        raise ValueError("test probability shape mismatch")
    if (not np.isfinite(test_probability).all() or (test_probability < 0).any()
            or not np.allclose(test_probability.sum(1), 1.0, atol=1e-8)):
        raise ValueError("test probabilities are invalid or not normalized")
    labels = np.asarray(class_order)[test_probability.argmax(1)]
    if not set(labels.tolist()) <= set(class_order):
        raise ValueError("predicted label is outside class order")
    output = sample.copy()
    output["SUBCLASS"] = labels
    path = Path(run_root) / "submissions" / f"submission_grade_{grade}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    output.to_csv(temporary, index=False)
    reloaded = pd.read_csv(temporary, encoding="utf-8")
    if list(reloaded.columns) != ["ID", "SUBCLASS"] or len(reloaded) != len(sample):
        raise ValueError("submission CSV roundtrip failed")
    if not np.array_equal(reloaded["ID"].astype(str).to_numpy(), sample["ID"].astype(str).to_numpy()):
        raise ValueError("submission ID order changed")
    if not set(reloaded["SUBCLASS"].tolist()) <= set(class_order):
        raise ValueError("submission contains an unknown class")
    os.replace(temporary, path)
    complete = Path(run_root) / "grades" / grade / "COMPLETE.json"
    atomic_write_json(complete, {
        "schema_version": "GRADED_SUBMISSION_COMPLETE_V1", "grade": grade,
        "submission_csv": str(path), "submission_sha256": sha256_file(path),
        "rows": len(output), "id_sha256": sequence_sha256(output["ID"].astype(str).to_numpy()),
        **receipt,
    })
    return path, read_json(complete)


def prepare(config_path, run_root):
    root = Path(__file__).parent
    config = read_json(config_path)
    if Path(run_root).resolve() != Path(config["run_root"]).resolve():
        raise ValueError("run root differs from runtime config")
    source_sha, source_rows = _source_identity(root)
    old_closure = read_json(config["old_source_closure_json"])
    verify_source_closure(Path(config["old_source_root"]), old_closure)
    if old_closure.get("composite_sha256") != OLD_SOURCE_SHA256:
        raise ValueError("old source closure differs from the immutable V1 source")
    resources = _resource_snapshot(config)
    atomic_write_json(Path(run_root) / "RESOURCE_PREFLIGHT.json", {
        "schema_version": "GRADED_RECOVERY_RESOURCE_PREFLIGHT_V1", **resources,
    })
    data = _load_training(config)
    report, a0, _ = _declarations(root)
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    outer0 = partitions["outer"][0]
    # This is the complete A0 integrity gate; it reads old train-only probabilities only.
    _load_old_base_oof(config, data, outer0, a0)
    _anchor_artifact_map(config)
    payload = {
        "schema_version": "GRADED_RECOVERY_PREPARED_V1", "source_sha256": source_sha,
        "runtime_config_sha256": sha256_file(config_path),
        "source_files": source_rows, "train_sha256": sha256_file(config["train_csv"]),
        "reference_sha256": sha256_file(config["reference_oof"]),
        "partition_sha256": canonical_sha256(partitions), "admission_sha256": report["admission_sha256"],
        "old_source_sha256": OLD_SOURCE_SHA256, "a0_candidates": 105,
        "test_read": False,
    }
    path = Path(run_root) / "PREPARED.json"
    if path.is_file() and read_json(path) != payload:
        raise ValueError("prepared recovery identity changed")
    atomic_write_json(path, payload)
    return payload


def run(config_path, run_root):
    root = Path(__file__).parent
    config = read_json(config_path)
    if Path(run_root).resolve() != Path(config["run_root"]).resolve():
        raise ValueError("run root differs from runtime config")
    prepared = read_json(Path(run_root) / "PREPARED.json")
    source_sha, _ = _source_identity(root)
    if source_sha != prepared["source_sha256"]:
        raise ValueError("recovery source changed after prepare")
    if sha256_file(config_path) != prepared["runtime_config_sha256"]:
        raise ValueError("runtime config changed after prepare")
    if (sha256_file(config["train_csv"]) != prepared["train_sha256"]
            or sha256_file(config["reference_oof"]) != prepared["reference_sha256"]):
        raise ValueError("train or reference input changed after prepare")
    data = _load_training(config)
    report, a0, declaration_by_id = _declarations(root)
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    if canonical_sha256(partitions) != prepared["partition_sha256"] or report["admission_sha256"] != prepared["admission_sha256"]:
        raise ValueError("recovery partition or admission changed after prepare")
    state_path = Path(run_root) / "STATE.json"
    events_path = Path(run_root) / "events.jsonl"
    state = read_json(state_path) if state_path.is_file() else {
        "schema_version": "GRADED_RECOVERY_STATE_V1", "status": "RUNNING",
        "current_grade": None, "completed_grades": [], "source_sha256": source_sha,
    }
    if state["source_sha256"] != source_sha:
        raise ValueError("state source identity changed")
    atomic_write_json(state_path, state)
    outer0 = partitions["outer"][0]
    outer0_indices, inner_folds, anchors, inner_c10, guard, old_probabilities = _load_old_inner_evidence(
        config, data, outer0, a0,
    )
    cumulative_new = []
    incumbent = None
    for grade in GRADE_ORDER:
        cumulative_new.extend(GRADE_ADDITIONS[grade])
        complete_path = Path(run_root) / "grades" / grade / "COMPLETE.json"
        selection_path = Path(run_root) / "grades" / grade / "selection.joblib"
        if complete_path.is_file():
            completed = _verify_stage_complete(run_root, grade)
            incumbent, _ = _freeze_selection(
                selection_path, grade,
                [row["candidate_id"] for row in a0] + cumulative_new,
            )
            if grade not in state["completed_grades"]:
                state["completed_grades"].append(grade)
            if completed["adaptive_full_oof_macro_f1"] >= float(config.get("target_macro_f1", 0.60)):
                state.update(status="COMPLETE", current_grade=None, stop_reason="TARGET_REACHED_AFTER_STAGE_CSV")
                atomic_write_json(state_path, state)
                return state
            continue
        state.update(status="RUNNING", current_grade=grade)
        atomic_write_json(state_path, state)
        append_jsonl(events_path, {"event": "GRADE_START", "grade": grade})
        new_declarations = [declaration_by_id[candidate_id] for candidate_id in cumulative_new]
        new_probabilities, failures = _new_stage_probabilities(
            config, run_root, data, outer0, inner_folds, new_declarations, source_sha,
        )
        if failures:
            state.update(
                status="PARTIAL", current_grade=grade, failed_candidates=failures,
                stop_reason="GRADE_CANDIDATES_INCOMPLETE",
            )
            atomic_write_json(Path(run_root) / "grades" / grade / "PARTIAL.json", {
                "schema_version": "GRADED_STAGE_PARTIAL_V1", "grade": grade,
                "failed_candidates": failures,
                "complete_candidates": sorted(new_probabilities),
                "next_grade_started": False,
            })
            atomic_write_json(state_path, state)
            append_jsonl(events_path, {
                "event": "GRADE_PARTIAL", "grade": grade, "failed_candidates": failures,
            })
            return state
        state["failed_candidates"] = []
        all_declarations = [*a0, *new_declarations]
        probabilities = {**old_probabilities, **new_probabilities}
        bank = _build_bank(all_declarations, probabilities)
        candidate_ids = [row["candidate_id"] for row in all_declarations]
        if selection_path.is_file():
            selection, selection_receipt = _freeze_selection(selection_path, grade, candidate_ids)
        else:
            selection = _select(anchors, bank, data["y_int"][outer0_indices], guard)
            if incumbent is not None and incumbent["selected"].macro_f1 > selection["selected"].macro_f1 + 1e-12:
                selection = incumbent
            selection, selection_receipt = _freeze_selection(
                selection_path, grade, candidate_ids, selection,
            )
        selection_sha = selection_receipt["selection_sha256"]
        # Selection is now frozen. Inference data is first opened below this line.
        test, sample = _load_inference(config, data)
        oof, test_probability, deployment = _deploy_stage(
            config, run_root, data, partitions, test, selection, inner_c10,
            outer0_indices, source_sha, declaration_by_id,
        )
        oof_score = float(f1_score(
            data["y_int"], oof.argmax(1), labels=np.arange(26), average="macro", zero_division=0,
        ))
        deployment_path = Path(run_root) / "grades" / grade / "DEPLOYMENT.json"
        deployment_receipt = {
            "schema_version": "GRADED_DEPLOYMENT_RECEIPT_V1", "grade": grade,
            "selection_sha256": selection_sha,
            "refit_plan": deployment["refit_plan"], "bias_receipt": deployment["bias_receipt"],
            "folds": 5, "test_fit": False, "test_used_for_selection": False,
        }
        if deployment_path.is_file() and read_json(deployment_path) != deployment_receipt:
            raise ValueError("deployment receipt changed on resume")
        atomic_write_json(deployment_path, deployment_receipt)
        predictions_path, predictions_receipt = _write_stage_predictions(
            run_root, grade, data, test, oof, test_probability, selection_sha,
        )
        submission_path, completed = _write_submission(
            run_root, grade, sample, test_probability, data["class_order"], {
                "selection_sha256": selection_sha,
                "selection_inner_macro_f1": float(selection["selected"].macro_f1),
                "adaptive_full_oof_macro_f1": oof_score,
                "historical_comparator": HISTORICAL_COMPARATOR,
                "candidate_count": len(all_declarations),
                "selected_recipe_id": selection["selected"].recipe_id,
                "selected_base_candidates": deployment["refit_plan"]["base_candidate_ids"],
                "test_sha256": config["test_sha256"],
                "sample_submission_sha256": config["sample_submission_sha256"],
                "predictions_npz": str(predictions_path),
                "predictions_sha256": predictions_receipt["npz_sha256"],
                "selection_used_test": False, "full_oof_used_for_recipe_selection": False,
                "stage_stopping_uses_full_oof": True,
                "validation_claim": "ADAPTIVE_OOF_NOT_INDEPENDENT_NESTED_VALIDATION",
                "website_submitted": False,
            },
        )
        incumbent = selection
        state["completed_grades"].append(grade)
        state.update(
            current_grade=None, latest_submission=str(submission_path),
            latest_inner_macro_f1=completed["selection_inner_macro_f1"],
            latest_adaptive_oof_macro_f1=oof_score,
        )
        atomic_write_json(state_path, state)
        append_jsonl(events_path, {
            "event": "GRADE_COMPLETE", "grade": grade, "submission": str(submission_path),
            "inner_macro_f1": completed["selection_inner_macro_f1"], "adaptive_oof_macro_f1": oof_score,
        })
        if completed["adaptive_full_oof_macro_f1"] >= float(config.get("target_macro_f1", 0.60)):
            state.update(status="COMPLETE", stop_reason="TARGET_REACHED_AFTER_STAGE_CSV")
            atomic_write_json(state_path, state)
            return state
    state.update(status="COMPLETE", current_grade=None, stop_reason="ALL_GRADES_COMPLETE")
    atomic_write_json(state_path, state)
    atomic_write_json(Path(run_root) / "COMPLETE.json", {
        "schema_version": "GRADED_RECOVERY_COMPLETE_V1", "status": "COMPLETE",
        "completed_grades": state["completed_grades"], "latest_submission": state.get("latest_submission"),
    })
    return state


def status(run_root):
    root = Path(run_root)
    if not root.exists():
        return {"status": "NOT_STARTED", "completed_grades": []}
    state = read_json(root / "STATE.json") if (root / "STATE.json").is_file() else {"status": "PREPARED"}
    grades = {grade: read_json(root / "grades" / grade / "COMPLETE.json")
              for grade in GRADE_ORDER if (root / "grades" / grade / "COMPLETE.json").is_file()}
    live = _runner_live(root)
    failure = read_json(root / "FAILURE.json") if (root / "FAILURE.json").is_file() else None
    result = {**state, "grades": grades, "failure": failure, "runner_live": live}
    if failure is not None and not live and state.get("status") != "COMPLETE":
        result["status"] = "FAILED"
    elif state.get("status") == "RUNNING" and not live:
        result["status"] = "INTERRUPTED_RESUMABLE"
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="graded-recovery")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True); prepare_parser.add_argument("--run-root", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True); run_parser.add_argument("--run-root", required=True)
    run_parser.add_argument("--start", action="store_true")
    status_parser = sub.add_parser("status"); status_parser.add_argument("--run-root", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(args.run_root), sort_keys=True)); return 0
    if args.command == "prepare":
        print(json.dumps(prepare(args.config, args.run_root), sort_keys=True)); return 0
    if not args.start:
        print(json.dumps({"status": "START_FLAG_REQUIRED"}, sort_keys=True)); return 2
    result = launch_detached(
        args.run_root, lambda: run(args.config, args.run_root), start=True,
    )
    print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == "STARTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
