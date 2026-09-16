"""Prepare exact train-only probability evidence after graded V2 has stopped."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

from combo_core import canonical_sha256, probability_sha256


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_stage_reproduction(reproduced, stored, *, tolerance=1e-12):
    reproduced = np.asarray(reproduced, dtype=np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    if reproduced.shape != stored.shape or not np.all(np.isfinite(reproduced)) or not np.all(np.isfinite(stored)):
        raise ValueError("latest V2 stage inner probability does not reproduce")
    max_abs = float(np.max(np.abs(reproduced - stored))) if stored.size else 0.0
    argmax_identical = bool(np.array_equal(reproduced.argmax(1), stored.argmax(1)))
    if max_abs > tolerance or not argmax_identical:
        raise ValueError("latest V2 stage inner probability does not reproduce")
    return {
        "max_abs": max_abs,
        "argmax_identical": argmax_identical,
        "reproduced_probability_sha256": probability_sha256(reproduced),
        "stored_probability_sha256": probability_sha256(stored),
    }


def _verify_npz(path, expected):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected):
            raise ValueError("partial combo evidence changed")
        for name, value in expected.items():
            if not np.array_equal(stored[name], value):
                raise ValueError("partial combo evidence changed")


def _save_or_verify_npz(path, expected):
    path = Path(path); expected = {name: np.asarray(value) for name, value in expected.items()}
    temporary = path.with_name(f".{path.name}.partial")
    if path.is_file():
        _verify_npz(path, expected)
        return _sha(path)
    if temporary.is_file():
        _verify_npz(temporary, expected)
    else:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **expected); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return _sha(path)


def _copy_or_verify(path, source):
    path, source = Path(path), Path(source)
    expected_sha = _sha(source)
    temporary = path.with_name(f".{path.name}.partial")
    if path.is_file():
        if _sha(path) != expected_sha: raise ValueError("partial combo evidence changed")
        return expected_sha
    if temporary.is_file():
        if _sha(temporary) != expected_sha: raise ValueError("partial combo evidence changed")
    else:
        shutil.copyfile(source, temporary)
    os.replace(temporary, path)
    return expected_sha


def _atomic_json(path, value):
    path = Path(path); temporary = path.with_name(f".{path.name}.partial")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.is_file():
        if path.read_bytes() != payload: raise ValueError("prepared combo evidence receipt changed")
        return
    if temporary.is_file():
        if temporary.read_bytes() != payload: raise ValueError("partial combo evidence receipt changed")
    else:
        with temporary.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _new_complete(v2, config, data, outer, inner_folds, declarations, source_sha):
    from base_bank import _lookup_key
    from base_receipts import canonical_sha256 as v2_sha, sequence_sha256
    from cache_store import ImmutableProbabilityCache, make_cache_key
    from source_closure import sha256_file

    cache = ImmutableProbabilityCache(Path(config["run_root"]) / "cache/outer_0/graded_base")
    outer_indices = np.asarray(outer["train_indices"], dtype=np.int64)
    partition_sha = v2_sha(outer)
    parser_sha = sha256_file(v2 / "vendor/mcmp_repr.py")
    complete = {}; fold_counts = {}
    for declaration in declarations:
        candidate_id = declaration["candidate_id"]
        probability = np.zeros((len(outer_indices), 26), dtype=np.float64)
        coverage = np.zeros(len(outer_indices), dtype=np.uint8)
        for fold in range(3):
            valid = np.flatnonzero(inner_folds == fold); train = np.flatnonzero(inner_folds != fold)
            prebindings = {
                "schema_version": "BASE_BANK_LOOKUP_V1", "candidate_id": candidate_id,
                "declaration_sha256": declaration["config_hash"], "fold": fold,
                "input_sha256": config["train_sha256"], "parser_sha256": parser_sha,
                "source_sha256": source_sha,
                "fit_ids_sha256": sequence_sha256(data["ids"][outer_indices][train]),
                "fit_y_sha256": sequence_sha256(data["y_label"][outer_indices][train]),
                "fit_groups_sha256": sequence_sha256(data["groups"][outer_indices][train]),
                "partition_sha256": v2_sha([partition_sha, fold, valid.tolist()]),
                "parameters_sha256": declaration["config_hash"],
            }
            key = cache.resolve_index(_lookup_key(prebindings))
            if key is None: continue
            receipt = json.loads((cache.artifact_path(key) / "COMPLETE.json").read_text())
            bindings = receipt.get("bindings", {})
            expected = {name: value for name, value in prebindings.items() if name.endswith("sha256")}
            if any(bindings.get(name) != value for name, value in expected.items()) or key != make_cache_key(bindings, supervised=True):
                raise ValueError(f"new candidate cache lineage mismatch: {candidate_id} fold {fold}")
            current, _ = cache._verify(key, bindings)
            if current.shape != (len(valid), 26):
                raise ValueError("new candidate probability shape mismatch")
            probability[valid] = current; coverage[valid] += 1
        fold_counts[candidate_id] = int(sum(np.all(coverage[inner_folds == fold] == 1) for fold in range(3)))
        if np.all(coverage == 1): complete[candidate_id] = probability
    return complete, fold_counts


def _anchors_and_guard(config, data, outer):
    from anchor_pipeline import C10_BLOCKS, anchor_model_recipes, build_anchor_oof
    from base_receipts import canonical_sha256 as v2_sha
    from feature_transforms import _mcmp_block_pair, _mcmp_rows
    from grade_runner import _anchor_artifact_map, _old_anchor_probability
    from support_guard import support_guard_mask

    outer_indices = np.asarray(outer["train_indices"], dtype=np.int64)
    local = {int(value): index for index, value in enumerate(outer_indices)}
    anchor_map = _anchor_artifact_map(config); recipes = anchor_model_recipes()
    anchors = {name: np.zeros((len(outer_indices), 26), dtype=np.float64) for name in (
        "A1_GEOMETRIC_GUARD", "A2_C10_BIAS_GEOMETRIC_GUARD",
    )}
    guard_oof = np.zeros(len(outer_indices), dtype=bool)
    for inner in outer["inner"]:
        fit = np.asarray(inner["train_indices"], dtype=np.int64)
        valid = np.asarray(inner["valid_indices"], dtype=np.int64)
        positions = np.asarray([local[int(value)] for value in valid])
        namespace = f"outer0-inner{inner['fold']}"; partition = v2_sha(inner)
        h1 = _old_anchor_probability(anchor_map, recipes["H1"], data, fit, f"{namespace}-H1", v2_sha([partition, "H1"]))
        c10 = _old_anchor_probability(anchor_map, recipes["C10"], data, fit, f"{namespace}-C10", v2_sha([partition, "C10"]))
        sub_oof = np.zeros((len(fit), 26), dtype=np.float64); coverage = np.zeros(len(fit), dtype=np.uint8)
        fit_local = {int(value): index for index, value in enumerate(fit)}
        for sub in inner["subinner"]:
            subfit = np.asarray(sub["train_indices"], dtype=np.int64)
            subvalid = np.asarray(sub["valid_indices"], dtype=np.int64)
            subpos = np.asarray([fit_local[int(value)] for value in subvalid])
            sub_oof[subpos] = _old_anchor_probability(
                anchor_map, recipes["C10"], data, subfit,
                f"outer0-inner{inner['fold']}-sub{sub['fold']}-C10", v2_sha(sub),
            ); coverage[subpos] += 1
        if not np.all(coverage == 1): raise ValueError("anchor sub-inner coverage mismatch")
        train_frame = data["frame"].iloc[fit][data["genes"]].reset_index(drop=True)
        valid_frame = data["frame"].iloc[valid][data["genes"]].reset_index(drop=True)
        train_rows, valid_rows = _mcmp_rows(train_frame, valid_frame, data["genes"])
        symbolic_train, symbolic_valid, _ = _mcmp_block_pair(
            train_rows, valid_rows, data["groups"][fit], C10_BLOCKS, 3,
        )
        guard, _ = support_guard_mask(symbolic_train, symbolic_valid); guard_oof[positions] = guard
        for name in anchors:
            anchors[name][positions] = build_anchor_oof(
                h1, c10, guard, sub_oof, data["y_int"][fit], branch=name,
            )["probability"]
    return anchors, guard_oof


def prerequisite_status(config):
    v2 = Path(config["v2_root"]); sys.path.insert(0, str(v2))
    from atomic_state import read_json
    from grade_runner import GRADE_ADDITIONS, GRADE_ORDER, _declarations, _load_old_base_oof
    from runner import _runner_live
    run_root = Path(config["v2_run_root"])
    result = {
        "schema_version": "COMBO_PREREQUISITE_STATUS_V1",
        "v2_runner_live": _runner_live(run_root), "complete_candidate_folds": {},
        "required_candidates": 124, "required_folds_each": 3,
    }
    if result["v2_runner_live"]:
        result.update(status="WAIT_V2_RUNNING", ready=False); return result
    v2_config = read_json(v2 / "runtime_config.recovery.windows.json")
    from workflow import _load_training
    from partition_contract import build_partitions
    data = _load_training(v2_config); _, a0, by_id = _declarations(v2)
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    outer = partitions["outer"][0]; outer_indices = np.asarray(outer["train_indices"])
    local = {int(value): index for index, value in enumerate(outer_indices)}
    inner_folds = np.empty(len(outer_indices), dtype=np.int16)
    for inner in outer["inner"]:
        inner_folds[[local[int(value)] for value in inner["valid_indices"]]] = int(inner["fold"])
    additions = [by_id[candidate_id] for grade in GRADE_ORDER[1:] for candidate_id in GRADE_ADDITIONS[grade]]
    prepared = read_json(run_root / "PREPARED.json")
    new, counts = _new_complete(v2, v2_config, data, outer, inner_folds, additions, prepared["source_sha256"])
    old = _load_old_base_oof(v2_config, data, outer, a0)
    if len(old) != 105:
        raise ValueError("old candidate cache verification did not return 105 candidates")
    result["complete_candidate_folds"] = {**{candidate_id: 3 for candidate_id in old}, **counts}
    result["complete_candidates"] = 105 + len(new)
    incomplete = sorted(name for name, count in result["complete_candidate_folds"].items() if count != 3)
    result["incomplete_candidates"] = incomplete
    result["ready"] = not incomplete and result["complete_candidates"] == 124
    result["status"] = "READY" if result["ready"] else "WAIT_INCOMPLETE_CANDIDATES"
    return result


def prepare_evidence(config, combo_run_root):
    status = prerequisite_status(config)
    combo_run_root = Path(combo_run_root); combo_run_root.mkdir(parents=True, exist_ok=True)
    (combo_run_root / "PREREQUISITE.json").write_text(json.dumps(status, indent=2) + "\n")
    if not status["ready"]:
        return status
    existing_receipt = combo_run_root / "EVIDENCE.json"
    if existing_receipt.is_file():
        existing = json.loads(existing_receipt.read_text())
        if (_sha(combo_run_root / "EVIDENCE.npz") != existing.get("evidence_sha256")
                or _sha(combo_run_root / "stage_selection.joblib") != existing.get("stage_selection_sha256")
                or _sha(combo_run_root / "stage_deployment.json") != existing.get("stage_deployment_sha256")):
            raise ValueError("prepared combo evidence changed")
        return {**status, **existing, "status": "EVIDENCE_PREPARED"}
    v2 = Path(config["v2_root"]); sys.path.insert(0, str(v2))
    from atomic_state import read_json
    from grade_runner import GRADE_ADDITIONS, GRADE_ORDER, _declarations, _load_old_base_oof, _source_identity
    from outer_procedure import apply_selected_recipe
    from source_closure import sha256_file
    from partition_contract import build_partitions
    from workflow import _load_training

    v2_config = read_json(v2 / "runtime_config.recovery.windows.json")
    data = _load_training(v2_config); admission, a0, by_id = _declarations(v2)
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    outer = partitions["outer"][0]; outer_indices = np.asarray(outer["train_indices"])
    local = {int(value): index for index, value in enumerate(outer_indices)}
    inner_folds = np.empty(len(outer_indices), dtype=np.int16)
    for inner in outer["inner"]:
        inner_folds[[local[int(value)] for value in inner["valid_indices"]]] = int(inner["fold"])
    prepared = read_json(Path(config["v2_run_root"]) / "PREPARED.json")
    current_source_sha, _ = _source_identity(v2)
    if current_source_sha != prepared["source_sha256"]:
        raise ValueError("current V2 source differs from PREPARED")
    if sha256_file(v2 / "runtime_config.recovery.windows.json") != prepared["runtime_config_sha256"]:
        raise ValueError("current V2 runtime config differs from PREPARED")
    additions = [by_id[candidate_id] for grade in GRADE_ORDER[1:] for candidate_id in GRADE_ADDITIONS[grade]]
    new, counts = _new_complete(v2, v2_config, data, outer, inner_folds, additions, prepared["source_sha256"])
    if len(new) != 19 or any(value != 3 for value in counts.values()):
        raise ValueError("all 19 new candidates must have three complete folds")
    old = _load_old_base_oof(v2_config, data, outer, a0)
    candidates = {**old, **new}
    if len(candidates) != 124: raise ValueError("candidate evidence count differs from 124")
    anchors, guard = _anchors_and_guard(v2_config, data, outer)
    latest = None
    for grade in reversed(GRADE_ORDER):
        root = Path(config["v2_run_root"]) / "grades" / grade
        if (root / "COMPLETE.json").is_file() and (root / "selection.joblib").is_file():
            latest = grade, root; break
    if latest is None: raise ValueError("no completed V2 grade selection")
    grade, grade_root = latest
    complete_receipt = read_json(grade_root / "COMPLETE.json")
    selection_receipt = read_json(grade_root / "SELECTION.json")
    selection_path = grade_root / "selection.joblib"
    selection_sha = _sha(selection_path)
    if (complete_receipt.get("selection_sha256") != selection_sha
            or selection_receipt.get("selection_sha256") != selection_sha):
        raise ValueError("latest V2 selection receipt/hash mismatch")
    selection = joblib.load(selection_path)
    stage_probability = np.asarray(selection["selected"].probability, dtype=np.float64)
    if stage_probability.shape != (len(outer_indices), 26): raise ValueError("stage inner probability shape mismatch")
    reproduced = apply_selected_recipe(
        selection["selected"], selection["candidate_graph"], anchors, candidates, guard,
    )
    reproduction = _verify_stage_reproduction(reproduced, stage_probability)
    candidate_ids = sorted(candidates)
    families = {row["candidate_id"]: row["family"] for row in admission["declarations"]}
    evidence_path = combo_run_root / "EVIDENCE.npz"
    expected_evidence = {
        "ids": np.asarray(data["ids"][outer_indices], dtype="U"), "y": data["y_int"][outer_indices],
        "groups": np.asarray(data["groups"][outer_indices], dtype="U"),
        "outer_folds": data["folds"][outer_indices], "inner_folds": inner_folds,
        "class_order": np.asarray(data["class_order"], dtype="U"), "guard": guard,
        "candidate_ids": np.asarray(candidate_ids, dtype="U"),
        "candidate_families": np.asarray([families[value] for value in candidate_ids], dtype="U"),
        "candidate_probabilities": np.stack([candidates[value] for value in candidate_ids]),
        "anchor_ids": np.asarray(sorted(anchors), dtype="U"),
        "anchor_probabilities": np.stack([anchors[value] for value in sorted(anchors)]),
        "stage_probability": stage_probability,
    }
    _save_or_verify_npz(evidence_path, expected_evidence)
    selection_copy = combo_run_root / "stage_selection.joblib"
    _copy_or_verify(selection_copy, selection_path)
    stage_deployment_path = combo_run_root / "stage_deployment.json"
    _copy_or_verify(stage_deployment_path, grade_root / "DEPLOYMENT.json")
    receipt = {
        "schema_version": "COMBO_EVIDENCE_RECEIPT_V1", "evidence_sha256": _sha(evidence_path),
        "ids_sha256": canonical_sha256(np.asarray(data["ids"][outer_indices], dtype="U").tolist()),
        "y_sha256": canonical_sha256(data["y_int"][outer_indices].tolist()),
        "groups_sha256": canonical_sha256(np.asarray(data["groups"][outer_indices], dtype="U").tolist()),
        "outer_folds_sha256": canonical_sha256(data["folds"][outer_indices].tolist()),
        "inner_folds_sha256": canonical_sha256(inner_folds.tolist()),
        "class_order_sha256": canonical_sha256(data["class_order"]),
        "candidate_count": 124, "candidate_folds_each": 3,
        "stage_grade": grade, "stage_selection_sha256": _sha(selection_copy),
        "stage_probability_sha256": probability_sha256(stage_probability),
        "stage_reproduced_probability_sha256": reproduction["reproduced_probability_sha256"],
        "stage_reproduction_max_abs": reproduction["max_abs"],
        "stage_reproduction_argmax_identical": reproduction["argmax_identical"],
        "stage_deployment_sha256": _sha(stage_deployment_path),
        "base_source_sha256": prepared["source_sha256"],
        "test_read": False, "server_score_used": False,
    }
    _atomic_json(combo_run_root / "EVIDENCE.json", receipt)
    return {**status, **receipt, "status": "EVIDENCE_PREPARED"}
