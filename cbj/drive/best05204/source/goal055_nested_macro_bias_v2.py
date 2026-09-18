"""Strict nested C10-only macro-F1 log-bias correction with fixed geometric pooling."""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_info, threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lab_metrics as L
import mcmp_repr as M
import cbj_repeat_seeds as Q
import goal055_tree_budget as T
import goal055_class_weight as W
import goal055_raw_annotation as R
import goal055_tfidf_logistic as F
import goal055_support_guard as V1
import goal055_geometric_support_guard as E


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-nested-macro-bias-v2":
        raise ValueError("Wrong nested macro-bias version")
    if config.get("seed") != 43 or config.get("outer_folds") != [0, 1, 2, 3, 4]:
        raise ValueError("Nested macro bias is restricted to seed43 outer folds")
    if config.get("inner_folds") != 3 or config.get("inner_seeds") != [4300, 4301, 4302, 4303, 4304]:
        raise ValueError("Nested macro-bias inner split protocol changed")
    if config.get("fit_budget") != 15 or config.get("weight_alpha") != 0.5:
        raise ValueError("Nested macro bias requires exactly15 inner C10 fits")
    if config.get("classifier_device") != "CPU" or config.get("threads") != 4:
        raise ValueError("Nested macro-bias CPU thread contract changed")
    if config.get("numeric_features") != 41 or config.get("numeric_block_weight") != "1/sqrt(41)":
        raise ValueError("Nested macro-bias numeric41 contract changed")
    if config.get("symbolic") != {"norm": "l2", "use_idf": True,
                                  "smooth_idf": True, "sublinear_tf": True}:
        raise ValueError("Nested macro-bias TF-IDF recipe changed")
    if config.get("logistic_model") != {"solver": "lbfgs", "penalty": "l2",
                                         "max_iter": 2000, "tol": 0.00001,
                                         "fit_intercept": True, "C": 10.0}:
        raise ValueError("Nested macro-bias C10 model changed")
    expected_bias = {
        "initial": "zeros26", "passes_max": 4, "delta_integer_min": -5,
        "delta_integer_max": 5, "delta_divisor": 10, "max_abs_centered_bias": 1.5,
        "l2_mean_penalty": 0.001, "strict_improvement_tolerance": 1e-12,
        "epsilon": 1e-300, "coordinate_order": "class_order_0_to_25",
    }
    if config.get("bias_optimizer") != expected_bias:
        raise ValueError("Nested macro-bias optimizer changed")
    if config.get("bias_objective") != "C10 inner OOF macro_f1 only":
        raise ValueError("Nested macro-bias objective must remain C10-only")
    if config.get("memory") != {"max_ram_gib": 20, "sparse_copies": 3,
                                "coefficient_copies": 30, "reserve_bytes": 1073741824,
                                "available_fraction": 0.8}:
        raise ValueError("Nested macro-bias memory admission contract changed")
    if (config.get("best_geometric_oof_sha256") !=
            "3b668367ca6d48f9c05f2a507458b6e0d530322722c99422980aba353722084a" or
            config.get("best_geometric_mac_oof_sha256") !=
            "a9666c597c02ca1c2b1d14b543110c97eb4777ee13b3c6d45a627167bf104596" or
            config.get("best_geometric_cross_platform_max_abs") != 1.1102230246251565e-16):
        raise ValueError("Nested macro-bias geometric cross-platform provenance changed")
    if any(config.get(key) is not False for key in
           ("test_read", "server_score_used", "submission", "outer_valid_tuning",
            "joint_h1_optimization", "calibration", "weight_search")):
        raise ValueError("Forbidden nested macro-bias data flow or search enabled")


def validate_inner_assignment(reference: dict, outer_indices, inner_assignment,
                              outer: int, expected: dict) -> list[dict]:
    outer_indices = np.asarray(outer_indices, dtype=np.int64)
    inner_assignment = np.asarray(inner_assignment, dtype=np.int16)
    canonical_outer_train = np.flatnonzero(reference["folds"] != outer)
    canonical_outer_valid = np.flatnonzero(reference["folds"] == outer)
    if not np.array_equal(outer_indices, canonical_outer_train):
        raise ValueError("Nested macro-bias outer training identity mismatch")
    if set(reference["groups"][outer_indices]) & set(reference["groups"][canonical_outer_valid]):
        raise ValueError("Nested macro-bias outer canonical group leakage")
    if inner_assignment.shape != (len(outer_indices),) or sorted(np.unique(inner_assignment).tolist()) != [0, 1, 2]:
        raise ValueError("Nested macro-bias inner assignment shape/labels invalid")
    if T.objsha(inner_assignment.tolist()) != expected["inner_assignment_sha256"]:
        raise ValueError("Nested macro-bias inner assignment object hash mismatch")
    if sha256_bytes(inner_assignment.astype("<i2", copy=False).tobytes()) != expected["inner_assignment_int16_le_bytes_sha256"]:
        raise ValueError("Nested macro-bias inner assignment byte hash mismatch")
    coverage = np.zeros(len(outer_indices), dtype=np.int8)
    receipts = []
    for inner in range(3):
        train_global = outer_indices[inner_assignment != inner]
        valid_global = outer_indices[inner_assignment == inner]
        coverage[inner_assignment == inner] += 1
        train_counts = np.bincount(reference["y"][train_global], minlength=26)
        valid_counts = np.bincount(reference["y"][valid_global], minlength=26)
        inner_expected = expected["inner"][str(inner)]
        values = {
            "train_rows": len(train_global), "valid_rows": len(valid_global),
            "train_groups": len(np.unique(reference["groups"][train_global])),
            "valid_groups": len(np.unique(reference["groups"][valid_global])),
            "train_indices_sha256": T.objsha(train_global.tolist()),
            "valid_indices_sha256": T.objsha(valid_global.tolist()),
            "train_ids_sha256": T.objsha(reference["ids"][train_global].tolist()),
            "valid_ids_sha256": T.objsha(reference["ids"][valid_global].tolist()),
            "train_groups_sha256": T.objsha(sorted(np.unique(reference["groups"][train_global]).tolist())),
            "valid_groups_sha256": T.objsha(sorted(np.unique(reference["groups"][valid_global]).tolist())),
            "train_class_counts": train_counts.tolist(), "valid_class_counts": valid_counts.tolist(),
            "train_class_counts_sha256": T.objsha(train_counts.tolist()),
            "valid_class_counts_sha256": T.objsha(valid_counts.tolist()),
        }
        if any(values[key] != inner_expected[key] for key in values):
            raise ValueError(f"Nested macro-bias frozen inner receipt mismatch: outer{outer}/inner{inner}")
        if (train_counts == 0).any() or (valid_counts == 0).any() or \
                set(reference["groups"][train_global]) & set(reference["groups"][valid_global]):
            raise ValueError(f"Nested macro-bias inner group/class admission failed: outer{outer}/inner{inner}")
        receipts.append({"inner": inner, **values, "train_indices": train_global,
                         "valid_indices": valid_global})
    if not np.all(coverage == 1):
        raise ValueError("Nested macro-bias inner validation coverage is not exact once")
    return receipts


def support_guard_helper_config(config: dict) -> dict:
    """Adapt only the fold-key spelling required by the frozen guard helper."""
    if config.get("outer_folds") != [0, 1, 2, 3, 4]:
        raise ValueError("Nested macro-bias outer folds cannot be adapted")
    adapted = dict(config)
    adapted["folds"] = list(config["outer_folds"])
    return adapted


def fit_inner_encoder(feature_rows, groups, train_indices, valid_indices,
                      blocks, min_support: int):
    encoder = M.Encoder(blocks, min_support)
    train = encoder.fit_transform([feature_rows[index] for index in train_indices],
                                  np.asarray(groups)[train_indices])
    valid = encoder.transform([feature_rows[index] for index in valid_indices])
    return encoder, train, valid


def write_gzip_json(path: Path, value) -> None:
    partial = Path(str(path) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    partial.replace(path)


def prepare_inner(config: dict, base_config: dict, feature_rows, raw_view,
                  reference: dict, outer: int, split: dict, out: Path):
    import cbj_summary_view as S

    inner = split["inner"]
    train_indices = split["train_indices"]
    valid_indices = split["valid_indices"]
    signature = T.objsha({
        "outer": outer, "inner": inner,
        "train_ids_sha256": split["train_ids_sha256"],
        "valid_ids_sha256": split["valid_ids_sha256"],
        "representation": base_config["representation"],
        "symbolic": config["symbolic"], "numeric_block_weight": config["numeric_block_weight"],
        "parser_sha256": config["mcmp_helper_sha256"],
    })
    job = Path(out) / "inner_transforms" / f"outer_{outer}" / f"inner_{inner}"
    if (job / "COMPLETE.json").is_file():
        T.verify_complete_receipt(job, signature)
        train = sparse.load_npz(job / "train_features.npz").tocsr()
        valid = sparse.load_npz(job / "valid_features.npz").tocsr()
        with np.load(job / "identity.npz", allow_pickle=False) as identity:
            if not np.array_equal(identity["train_indices"], train_indices) or \
                    not np.array_equal(identity["valid_indices"], valid_indices):
                raise ValueError("Nested macro-bias reused inner identity mismatch")
        feature_order = T.read_json(job / "feature_order.json")["feature_order"]
        memory = inner_memory_admission(config, train, valid, len(reference["class_order"]))
        return {"train": train, "valid": valid, "train_indices": train_indices,
                "valid_indices": valid_indices, "feature_order": feature_order,
                "transform_receipt_sha256": T.sha(job / "COMPLETE.json"), "memory": memory}
    if job.exists() and any(job.iterdir()):
        raise ValueError(f"Incomplete nested inner transform exists: {job}")
    job.mkdir(parents=True, exist_ok=True)
    encoder, matrix_train, matrix_valid = fit_inner_encoder(
        feature_rows, reference["groups"], train_indices, valid_indices,
        base_config["representation"]["blocks"], base_config["representation"]["min_group_support"])
    names = encoder.names
    original_global = [name for name in names if name.startswith("global|")]
    if original_global != ["global|missing_gene_count", "global|mutation_gene_count",
                           "global|observed_gene_count"]:
        raise ValueError("Nested macro-bias original global3 contract changed")
    take = F.symbolic_indices(names)
    if len(take) != len(names) - 3:
        raise ValueError("Nested macro-bias symbolic selection did not exclude exactly global3")
    global_train, global_names, _, _ = S.build_summary_views(
        matrix_train, names, {"placeholder": {"__not_a_gene__"}}, len(base_config["source_genes"]))
    global_valid, valid_global_names, _, _ = S.build_summary_views(
        matrix_valid, names, {"placeholder": {"__not_a_gene__"}}, len(base_config["source_genes"]))
    if global_names != valid_global_names or len(global_names) != 29:
        raise ValueError("Nested macro-bias global29 order changed")
    numeric_train = np.hstack([global_train.toarray(), raw_view[train_indices].toarray()])
    numeric_valid = np.hstack([global_valid.toarray(), raw_view[valid_indices].toarray()])
    transformed = F.fit_feature_blocks(matrix_train[:, take], matrix_valid[:, take],
                                       numeric_train, numeric_valid)
    feature_order = [names[index] for index in take] + global_names + list(R.RAW_NAMES)
    if transformed["train"].shape[1] != len(feature_order) or len(feature_order) < 41:
        raise ValueError("Nested macro-bias transformed feature order invalid")
    memory = inner_memory_admission(config, transformed["train"], transformed["valid"],
                                    len(reference["class_order"]))
    sparse.save_npz(job / "train_features.npz", transformed["train"], compressed=True)
    sparse.save_npz(job / "valid_features.npz", transformed["valid"], compressed=True)
    write_gzip_json(job / "encoder.json.gz", encoder.to_dict())
    T.save_npz(job / "preprocessor.npz", idf=transformed["idf"],
               scaler_mean=transformed["scaler_mean"], scaler_scale=transformed["scaler_scale"],
               symbolic_indices=np.asarray(take, dtype=np.int32))
    T.save_npz(job / "identity.npz", train_indices=train_indices, valid_indices=valid_indices)
    T.write_json(job / "feature_order.json", {"feature_order": feature_order,
                 "symbolic_features": [names[index] for index in take],
                 "numeric_features": global_names + list(R.RAW_NAMES)})
    review = {
        "outer": outer, "inner": inner, "train_rows": len(train_indices),
        "valid_rows": len(valid_indices), "features": len(feature_order),
        "encoder_min_support": encoder.min_support,
        "train_ids_sha256": split["train_ids_sha256"],
        "valid_ids_sha256": split["valid_ids_sha256"],
        "feature_order_sha256": T.objsha(feature_order),
        "encoder_fit_scope": "inner-training groups only",
        "tfidf_fit_scope": "inner-training symbolic matrix only",
        "scaler_fit_scope": "inner-training numeric41 only",
        "memory_admission_at_transform": memory,
        "validation_vocabulary_idf_scaler_used": False, "test_read": False,
    }
    T.write_json(job / "review.json", review)
    files = {name: T.sha(job / name) for name in (
        "train_features.npz", "valid_features.npz", "encoder.json.gz", "preprocessor.npz",
        "identity.npz", "feature_order.json", "review.json")}
    T.write_json(job / "COMPLETE.json", {"signature": signature, "files": files})
    T.verify_complete_receipt(job, signature)
    return {"train": transformed["train"], "valid": transformed["valid"],
            "train_indices": train_indices, "valid_indices": valid_indices,
            "feature_order": feature_order,
            "transform_receipt_sha256": T.sha(job / "COMPLETE.json"), "memory": memory}


def sparse_bytes(matrix) -> int:
    matrix = matrix.tocsr()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def inner_memory_admission(config: dict, train_matrix, valid_matrix, classes: int) -> dict:
    memory = config["memory"]
    sparse_total = sparse_bytes(train_matrix) + sparse_bytes(valid_matrix)
    coefficient_bytes = int(classes * train_matrix.shape[1] * 8)
    process_rss = int(__import__("psutil").Process().memory_info().rss)
    available = int(__import__("psutil").virtual_memory().available)
    estimated = int(process_rss + memory["sparse_copies"] * sparse_total +
                    memory["coefficient_copies"] * coefficient_bytes + memory["reserve_bytes"])
    limit = int(min(memory["max_ram_gib"] * 2**30, available * memory["available_fraction"]))
    if estimated >= limit:
        raise ValueError("Nested macro-bias inner CPU memory admission failed")
    return {"process_rss_bytes": process_rss, "combined_sparse_bytes": sparse_total,
            "coefficient_bytes": coefficient_bytes, "estimated_peak_bytes": estimated,
            "available_ram_bytes": available, "admission_limit_bytes": limit,
            "admitted": True}


def completed_inner_fit(job: Path, signature: str):
    if not (job / "COMPLETE.json").is_file():
        return None
    T.verify_complete_receipt(job, signature)
    with np.load(job / "valid_prob.npz", allow_pickle=False) as stored:
        return np.asarray(stored["prob"], dtype=np.float64)


def train_inner(config: dict, reference: dict, cache: dict, outer: int,
                inner: int, out: Path):
    train_indices = cache["train_indices"]; valid_indices = cache["valid_indices"]
    weights, weight_receipt = W.fold_local_sample_weights(
        reference["y"][train_indices], len(reference["class_order"]), config["weight_alpha"])
    signature = T.objsha({
        "config_model": config["logistic_model"], "outer": outer, "inner": inner,
        "transform_receipt_sha256": cache["transform_receipt_sha256"],
        "fit_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
        "fit_y_sha256": T.objsha(reference["y"][train_indices].tolist()),
        "weights_sha256": T.objsha(weights.tolist()),
    })
    job = Path(out) / "inner_models" / f"outer_{outer}" / f"inner_{inner}"
    reused = completed_inner_fit(job, signature)
    if reused is not None:
        T.emit(out, "NESTED_BIAS_INNER_FIT_REUSED", outer=outer, inner=inner)
        return reused, True
    if job.exists() and any(job.iterdir()):
        raise ValueError(f"Incomplete nested inner model exists: {job}")
    partial = job.parent / f".inner_{inner}.{os.getpid()}.partial"
    partial.mkdir(parents=True, exist_ok=False)
    memory_at_fit = inner_memory_admission(config, cache["train"], cache["valid"],
                                           len(reference["class_order"]))
    model = LogisticRegression(**config["logistic_model"])
    T.emit(out, "NESTED_BIAS_INNER_CPU_FIT_START", outer=outer, inner=inner,
           rows=len(train_indices), features=cache["train"].shape[1], C=10.0)
    started = time.perf_counter()
    with threadpool_limits(limits=config["threads"]):
        pools = threadpool_info()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(cache["train"], reference["y"][train_indices], sample_weight=weights)
    seconds = time.perf_counter() - started
    F.require_convergence(model, captured, config["logistic_model"]["max_iter"])
    if model.classes_.tolist() != list(range(len(reference["class_order"]))):
        raise ValueError("Nested macro-bias inner class order changed")
    train_probability = np.asarray(model.predict_proba(cache["train"]), dtype=np.float64)
    probability = np.asarray(model.predict_proba(cache["valid"]), dtype=np.float64)
    F.validate_oof_probability(train_probability, len(train_indices), len(reference["class_order"]))
    F.validate_oof_probability(probability, len(valid_indices), len(reference["class_order"]))
    joblib.dump(model, partial / "model.joblib", compress=3)
    T.save_npz(partial / "valid_prob.npz", valid_indices=valid_indices, prob=probability)
    restored = joblib.load(partial / "model.joblib")
    probe = restored.predict_proba(cache["valid"][:min(16, len(valid_indices))])
    reload_max = float(np.max(np.abs(probe - probability[:len(probe)])))
    if reload_max > config["reload_tolerance"]:
        raise ValueError("Nested macro-bias inner model replay mismatch")
    train_f1 = float(f1_score(reference["y"][train_indices], train_probability.argmax(1),
                              labels=np.arange(26), average="macro", zero_division=0))
    valid_f1 = float(f1_score(reference["y"][valid_indices], probability.argmax(1),
                              labels=np.arange(26), average="macro", zero_division=0))
    T.write_json(partial / "review.json", {
        "outer": outer, "inner": inner, "fit_seconds": seconds,
        "requested_parameters": config["logistic_model"], "classes": model.classes_.tolist(),
        "n_iter": model.n_iter_.tolist(), "convergence_status": "CONVERGED",
        "warnings": [str(item.message) for item in captured],
        "fit_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
        "fit_y_sha256": T.objsha(reference["y"][train_indices].tolist()),
        "validation_ids_sha256": T.objsha(reference["ids"][valid_indices].tolist()),
        "transform_receipt_sha256": cache["transform_receipt_sha256"],
        "feature_order_sha256": T.objsha(cache["feature_order"]),
        "weight_receipt": weight_receipt, "train_macro_f1": train_f1,
        "valid_macro_f1": valid_f1, "train_valid_gap": train_f1 - valid_f1,
        "reload_max_abs": reload_max, "effective_threadpools": pools,
        "memory_admission_before_fit": memory_at_fit,
        "validation_set_passed_to_fit": False, "classifier_device": "CPU",
        "test_read": False, "server_score_used": False,
    })
    files = {name: T.sha(partial / name) for name in ("model.joblib", "valid_prob.npz", "review.json")}
    T.write_json(partial / "COMPLETE.json", {"signature": signature, "files": files})
    T.verify_complete_receipt(partial, signature)
    if job.exists():
        raise ValueError(f"Nested inner model destination appeared during fit: {job}")
    partial.replace(job)
    T.emit(out, "NESTED_BIAS_INNER_CPU_FIT_COMPLETE", outer=outer, inner=inner,
           fit_seconds=seconds, valid_macro_f1=valid_f1)
    return probability, False


def bias_state(probability, y, bias, penalty: float, epsilon: float):
    logits = np.log(np.maximum(np.asarray(probability, dtype=np.float64), epsilon)) + bias
    prediction = logits.argmax(axis=1)
    macro = float(f1_score(y, prediction, labels=np.arange(probability.shape[1]),
                           average="macro", zero_division=0))
    norm = float(np.mean(np.square(bias)))
    return {"objective": macro - penalty * norm, "macro_f1": macro,
            "mean_bias_squared": norm, "prediction": prediction}


def candidate_better(candidate, incumbent, tolerance: float) -> bool:
    for key, higher in (("objective", True), ("macro_f1", True),
                        ("mean_bias_squared", False), ("abs_delta", False),
                        ("delta", False)):
        left, right = candidate[key], incumbent[key]
        if abs(left - right) <= tolerance:
            continue
        return left > right if higher else left < right
    return False


def fit_c10_macro_bias(inner_probability, inner_y, optimizer: dict):
    probability = np.asarray(inner_probability, dtype=np.float64)
    y = np.asarray(inner_y, dtype=np.int64)
    F.validate_oof_probability(probability, len(y), 26)
    bias = np.zeros(26, dtype=np.float64)
    penalty = optimizer["l2_mean_penalty"]; tolerance = optimizer["strict_improvement_tolerance"]
    epsilon = optimizer["epsilon"]; bound = optimizer["max_abs_centered_bias"]
    history = []
    current = bias_state(probability, y, bias, penalty, epsilon)
    initial = {key: current[key] for key in ("objective", "macro_f1", "mean_bias_squared")}
    attempted_passes = 0
    for pass_index in range(optimizer["passes_max"]):
        attempted_passes += 1
        commits = 0
        for coordinate in range(26):
            incumbent = {**current, "delta": 0.0, "abs_delta": 0.0, "bias": bias.copy()}
            best = incumbent
            for integer_delta in range(optimizer["delta_integer_min"],
                                       optimizer["delta_integer_max"] + 1):
                delta = integer_delta / optimizer["delta_divisor"]
                candidate_bias = bias.copy(); candidate_bias[coordinate] += delta
                candidate_bias -= candidate_bias.mean()
                if np.max(np.abs(candidate_bias)) > bound + tolerance:
                    continue
                state = bias_state(probability, y, candidate_bias, penalty, epsilon)
                candidate = {**state, "delta": delta, "abs_delta": abs(delta),
                             "bias": candidate_bias}
                if candidate_better(candidate, best, tolerance):
                    best = candidate
            if best["objective"] > current["objective"] + tolerance:
                previous = current["objective"]
                bias = best["bias"]
                current = {key: best[key] for key in
                           ("objective", "macro_f1", "mean_bias_squared", "prediction")}
                history.append({"pass": pass_index, "coordinate": coordinate,
                                "delta": best["delta"], "objective_before": previous,
                                "objective_after": current["objective"],
                                "macro_f1_after": current["macro_f1"],
                                "mean_bias_squared_after": current["mean_bias_squared"]})
                commits += 1
        if commits == 0:
            break
    if (abs(float(bias.mean())) > 1e-12 or np.max(np.abs(bias)) > bound + tolerance or
            any(item["objective_after"] <= item["objective_before"] + tolerance for item in history)):
        raise ValueError("Nested macro-bias optimizer invariant failed")
    final = bias_state(probability, y, bias, penalty, epsilon)
    return bias, {"initial": initial,
                  "final": {key: final[key] for key in ("objective", "macro_f1", "mean_bias_squared")},
                  "history": history, "commits": len(history),
                  "passes_attempted": attempted_passes,
                  "centered_bias": bias.tolist(), "C10_only_objective": True,
                  "outer_validation_labels_accepted": False}


def apply_log_bias(probability, bias, epsilon: float = 1e-300):
    probability = np.asarray(probability, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if probability.ndim != 2 or bias.shape != (probability.shape[1],):
        raise ValueError("Nested macro-bias application shape mismatch")
    logits = np.log(np.maximum(probability, epsilon)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits); result /= result.sum(axis=1, keepdims=True)
    return result


def apply_outer_biases(c10_probability, folds, biases, epsilon: float = 1e-300):
    probability = np.asarray(c10_probability, dtype=np.float64)
    folds = np.asarray(folds)
    result = np.full_like(probability, np.nan)
    coverage = np.zeros(len(folds), dtype=np.int8)
    for outer in range(5):
        take = folds == outer
        result[take] = apply_log_bias(probability[take], np.asarray(biases[str(outer)]), epsilon)
        coverage[take] += 1
    if not np.all(coverage == 1):
        raise ValueError("Nested macro-bias outer application coverage failed")
    return result


def run_signature(config: dict, h1_oof: Path, c10_oof: Path):
    return T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                     "h1_oof_sha256": T.sha(h1_oof), "c10_oof_sha256": T.sha(c10_oof)})


def expected_root_files():
    endpoints = ("geometric_support_guard_best", "bias_corrected_c10", "bias_corrected_geometric_guard")
    files = ["CONTRACT.json", "environment.json", "ADMISSION.json", "INNER_ASSIGNMENTS.npz",
             "INNER_ASSIGNMENT_RECEIPT.json", "BASELINE_REPLAY.json", "BIAS_RECEIPTS.json",
             "NESTED_MACRO_BIAS_REVIEW.json", "nested_macro_bias_scoreboard.csv", "OOF_READBACK.json"]
    for name in endpoints:
        files += [f"{name}_oof.npz", f"{name}_per_class.csv"]
    files += [f"outer_{outer}_inner_oof.npz" for outer in range(5)]
    return files


def completed_run(out: Path, signature: str):
    marker = Path(out) / "NESTED_MACRO_BIAS_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker); files = receipt.get("files")
    expected_models = [f"inner_models/outer_{outer}/inner_{inner}"
                       for outer in range(5) for inner in range(3)]
    expected_transforms = [f"inner_transforms/outer_{outer}/inner_{inner}"
                           for outer in range(5) for inner in range(3)]
    outer_oofs = receipt.get("outer_inner_oof_files")
    if (receipt.get("signature") != signature or not isinstance(files, dict) or
            set(files) != set(expected_root_files()) or
            receipt.get("model_receipts") != expected_models or
            receipt.get("transform_receipts") != expected_transforms or
            not isinstance(outer_oofs, dict) or set(outer_oofs) != {str(i) for i in range(5)} or
            receipt.get("new_cpu_fits", -1) + receipt.get("fits_reused", -1) != 15):
        raise ValueError("Malformed nested macro-bias completion receipt")
    for name, digest in files.items():
        path = Path(out) / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Nested macro-bias completion integrity failure: {path}")
    for relative in expected_models + expected_transforms:
        T.verify_complete_receipt(Path(out) / relative)
    for outer, digest in outer_oofs.items():
        path = Path(out) / f"outer_{outer}_inner_oof.npz"
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Nested macro-bias outer inner-OOF integrity failure: {outer}")
    return True


def preflight(config: dict, base_config_path: Path, inputs: Path, repeat_root: Path,
              h1_oof: Path, c10_oof: Path, h1_run: Path, tfidf_run: Path,
              design_receipt_path: Path, out: Path):
    validate_config(config)
    helpers = {"tree_helper_sha256": Path(T.__file__), "weight_helper_sha256": Path(W.__file__),
               "raw_helper_sha256": Path(R.__file__), "tfidf_helper_sha256": Path(F.__file__),
               "support_helper_sha256": Path(V1.__file__), "geometric_helper_sha256": Path(E.__file__),
               "mcmp_helper_sha256": Path(M.__file__), "repeat_helper_sha256": Path(Q.__file__),
               "summary_helper_sha256": PROJECT_ROOT / "cbj_summary_view.py",
               "lab_metrics_sha256": Path(L.__file__)}
    for key, path in helpers.items():
        if T.sha(path) != config.get(key):
            raise ValueError(f"Pinned nested macro-bias helper mismatch: {key}")
    T.verify_named_hashes(T.PROJECT_ROOT, config["frozen_source_hashes"], "frozen source")
    if T.sha(base_config_path) != config["base_config_sha256"]:
        raise ValueError("Nested macro-bias base config hash mismatch")
    T.verify_named_hashes(inputs, config["input_hashes"], "training input")
    T.verify_named_hashes(repeat_root, config["repeat_root_hashes"], "repeat artifact")
    if T.sha(design_receipt_path) != config["design_receipt_sha256"]:
        raise ValueError("Nested macro-bias design receipt hash mismatch")
    design = T.read_json(design_receipt_path)
    if design.get("status") != "DESIGN_ONLY_NO_CLASSIFIER_FITS" or design.get("new_classifier_fits_executed_for_receipt") != 0:
        raise ValueError("Nested macro-bias design receipt provenance changed")
    source_paths = {"h1_oof_sha256": h1_oof, "c10_oof_sha256": c10_oof,
                    "h1_complete_sha256": h1_run / "H1_COMPLETE.json",
                    "tfidf_complete_sha256": tfidf_run / "TFIDF_LOGISTIC_COMPLETE.json"}
    for key, path in source_paths.items():
        if T.sha(path) != config[key]:
            raise ValueError(f"Nested macro-bias source hash mismatch: {key}")
    h1 = L.load_oof(h1_oof); c10 = L.load_oof(c10_oof); V1.validate_exact_alignment(h1, c10)
    reference = L.load_oof(repeat_root / "seed_43" / "mrmr_global_oof.npz")
    V1.validate_exact_alignment(reference, h1)
    base_config = T.read_json(base_config_path)
    versions = {name: importlib.metadata.version(name)
                for name in base_config["packages"]}
    if versions != base_config["packages"] or versions.get("scikit-learn") != "1.8.0":
        raise ValueError("Nested macro-bias Windows sklearn/base environment changed")
    if (platform.python_version() != design["environment"]["python"] or
            versions["numpy"] != design["environment"]["numpy"]):
        raise ValueError("Nested macro-bias design/runtime environment mismatch")
    raw = pd.read_csv(inputs / base_config["inputs"]["train"]["filename"], dtype=str)
    if raw.columns.tolist() != ["ID", "SUBCLASS"] + base_config["source_genes"]:
        raise ValueError("Nested macro-bias raw schema/gene order mismatch")
    if raw.ID.tolist() != reference["ids"].tolist() or \
            raw.SUBCLASS.tolist() != reference["class_order"][reference["y"]].tolist():
        raise ValueError("Nested macro-bias raw ID/target identity mismatch")
    feature_rows, parser_stats = M.build_feature_rows(raw, base_config["source_genes"])
    raw_view, raw_names, raw_stats = R.build_raw_annotation_view(raw[base_config["source_genes"]],
                                                                base_config["source_genes"])
    del raw
    if raw_names != list(R.RAW_NAMES):
        raise ValueError("Nested macro-bias raw12 order mismatch")
    R.validate_raw_stats(raw_stats, config["expected_raw_stats"])
    assignments = {}; split_receipts = {}
    for outer in config["outer_folds"]:
        outer_indices = np.flatnonzero(reference["folds"] != outer)
        inner_assignment = Q.assign_group_folds(reference["y"][outer_indices],
                                                reference["groups"][outer_indices],
                                                config["inner_seeds"][outer], 3)
        splits = validate_inner_assignment(reference, outer_indices, inner_assignment,
                                           outer, design["outer"][str(outer)])
        assignments[f"outer_{outer}_indices"] = outer_indices
        assignments[f"outer_{outer}_inner_folds"] = inner_assignment
        split_receipts[str(outer)] = [{key: value for key, value in split.items()
                                       if key not in ("train_indices", "valid_indices")}
                                      for split in splits]
    signature = run_signature(config, h1_oof, c10_oof)
    out.mkdir(parents=True, exist_ok=True)
    contract = out / "CONTRACT.json"
    if contract.exists():
        if T.read_json(contract).get("signature") != signature:
            raise ValueError("Nested macro-bias output belongs to another configuration")
    else:
        unexpected = sorted(path.name for path in out.iterdir()
                            if path.name not in {"launch.json", "execution.log"})
        if unexpected:
            raise ValueError("Nested macro-bias output contains unrelated artifacts")
        T.write_json(contract, {"signature": signature, "config": config,
                     "runner_sha256": T.sha(Path(__file__)),
                     "scope": "strict nested C10-only macro-bias;15 inner fits; one corrected candidate"})
    T.save_npz(out / "INNER_ASSIGNMENTS.npz", **assignments)
    T.write_json(out / "INNER_ASSIGNMENT_RECEIPT.json", {
        "environment": design["environment"], "splitter": design["splitter"],
        "outer": split_receipts, "all_15_frozen_before_fits": True,
        "all_groups_disjoint_and_all_26_classes": True, "fit_count_at_freeze": 0})
    guard, guard_evidence = E.reconstruct_guard(
        support_guard_helper_config(config), repeat_root / "seed_43" / "prepared", h1)
    best = E.guarded_geometric_pool(h1["prob"], c10["prob"], guard, 1e-300)
    baseline_path = out / "geometric_support_guard_best_oof.npz"
    L.save_npz(baseline_path, **{key: reference[key]
               for key in ("ids", "y", "groups", "folds", "class_order")}, prob=best)
    baseline_hash = T.sha(baseline_path)
    if baseline_hash != config["best_geometric_oof_sha256"] or \
            L.metrics(reference, best)["macro_f1"] != config["best_geometric_macro_f1"]:
        raise ValueError("Nested macro-bias geometric baseline replay mismatch")
    T.write_json(out / "BASELINE_REPLAY.json", {
        "source_h1_oof_sha256": config["h1_oof_sha256"],
        "source_c10_oof_sha256": config["c10_oof_sha256"],
        "replayed_best_oof_sha256": baseline_hash,
        "Mac_best_oof_sha256": config["best_geometric_mac_oof_sha256"],
        "audited_cross_platform_max_abs": config["best_geometric_cross_platform_max_abs"],
        "macro_f1": config["best_geometric_macro_f1"], "guarded_rows": int(guard.sum()),
        "guard_evidence": guard_evidence, "Mac_complete_receipt_assumed_on_Windows": False,
    })
    available = __import__("psutil").virtual_memory().available
    if config["estimated_peak_bytes"] >= min(config["memory"]["max_ram_gib"] * 2**30,
                                              available * config["memory"]["available_fraction"]):
        raise ValueError("Nested macro-bias design CPU memory admission failed")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": versions, "classifier_device": "CPU", "threads": 4}
    T.write_json(out / "environment.json", environment)
    admission = {"rows": len(reference["ids"]), "classes": len(reference["class_order"]),
                 "canonical_groups": len(np.unique(reference["groups"])), "fit_budget": 15,
                 "inner_folds_frozen": 15, "all_inner_splits_have_26_classes": True,
                 "all_inner_groups_disjoint": True, "classifier_device": "CPU", "threads": 4,
                 "estimated_runtime": "UNMEASURED 3-8 minutes CPU",
                 "design_estimated_peak_bytes": config["estimated_peak_bytes"],
                 "available_ram_bytes": available, "max_ram_bytes": base_config["runtime"]["max_ram_gib"] * 2**30,
                 "outer_validation_used_for_bias_fit": False, "test_read": False}
    T.write_json(out / "ADMISSION.json", admission)
    T.emit(out, "NESTED_BIAS_PREFLIGHT_COMPLETE", inner_folds_frozen=15,
           fit_budget=15, sklearn=versions["scikit-learn"])
    return reference, h1, c10, best, guard, feature_rows, raw_view, base_config, assignments, environment, signature


def save_endpoint_oofs_allow_equal(out: Path, reference: dict,
                                   endpoints: dict[str, np.ndarray]):
    """Bind and read back every endpoint independently; equality is a valid null result."""
    receipt = {}
    for name, endpoint_probability in endpoints.items():
        probability = E.validate_probability(endpoint_probability, len(reference["ids"]),
                                             len(reference["class_order"]))
        path = Path(out) / f"{name}_oof.npz"
        L.save_npz(path, **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=probability)
        with np.load(path, allow_pickle=False) as raw:
            if not np.array_equal(raw["prob"], probability):
                raise ValueError(f"Raw nested macro-bias endpoint export mismatch: {name}")
            for key in ("ids", "y", "groups", "folds", "class_order"):
                if not np.array_equal(raw[key], reference[key]):
                    raise ValueError(f"Nested macro-bias endpoint identity mismatch: {name}/{key}")
        loaded = L.load_oof(path); V1.validate_exact_alignment(reference, loaded)
        normalized_max = float(np.max(np.abs(loaded["prob"] - probability)))
        source_score = L.metrics(reference, probability)["macro_f1"]
        stored_score = L.metrics(reference, loaded["prob"])["macro_f1"]
        if normalized_max > 1e-12 or source_score != stored_score:
            raise ValueError(f"Nested macro-bias endpoint score/readback mismatch: {name}")
        receipt[name] = {"oof_sha256": T.sha(path), "normalized_max_abs": normalized_max,
                         "source_macro_f1": source_score, "stored_macro_f1": stored_score,
                         "may_equal_another_endpoint_if_bias_is_null": True}
    return receipt


def report(config: dict, reference: dict, best, corrected_c10, corrected_candidate,
           bias_receipts: dict, out: Path, executed: int, reused: int):
    endpoints = {"geometric_support_guard_best": best,
                 "bias_corrected_c10": corrected_c10,
                 "bias_corrected_geometric_guard": corrected_candidate}
    scores = {name: L.metrics(reference, probability) for name, probability in endpoints.items()}
    readback = save_endpoint_oofs_allow_equal(out, reference, endpoints)
    rows = []
    for name, score in scores.items():
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])},
                     "probability_provenance": ("normalized decision-adjusted; not calibrated"
                                                if "bias_corrected" in name else "frozen fixed pool")})
    pd.DataFrame(rows).to_csv(out / "nested_macro_bias_scoreboard.csv", index=False)
    comparisons = {
        "corrected_candidate_vs_best": {
            "delta_macro_f1": scores["bias_corrected_geometric_guard"]["macro_f1"] -
                              scores["geometric_support_guard_best"]["macro_f1"],
            **L.changes(reference["y"], corrected_candidate, best),
            "bootstrap": L.bootstrap(reference, corrected_candidate, best,
                                     config["bootstrap_resamples"], 43),
        },
        "corrected_c10_vs_frozen_c10_diagnostic": {
            "delta_macro_f1": scores["bias_corrected_c10"]["macro_f1"] - config["frozen_c10_macro_f1"]
        },
    }
    T.write_json(out / "BIAS_RECEIPTS.json", bias_receipts)
    T.write_json(out / "OOF_READBACK.json", {"endpoints": readback})
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_NESTED_C10_MACRO_BIAS_COMPLETE",
        "bias_objective": "C10 inner OOF macro_f1 only", "joint_H1_optimization": False,
        "scores": {name: {key: value for key, value in score.items() if key != "per_class"}
                   for name, score in scores.items()},
        "comparisons": comparisons, "fits_executed": executed, "fits_reused": reused,
        "fit_budget": 15, "corrected_probability_claim": "normalized decision-adjusted; not calibrated",
        "outer_validation_used_for_bias_fit": False, "automatic_promotion": False,
        "test_read": False, "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "NESTED_MACRO_BIAS_REVIEW.json", review)


def run(args):
    config = T.read_json(Path(args.config)); out = Path(args.out).resolve()
    h1_oof = Path(args.h1_oof).resolve(); c10_oof = Path(args.c10_oof).resolve()
    signature = run_signature(config, h1_oof, c10_oof)
    if completed_run(out, signature):
        print(json.dumps({"event": "COMPLETED_NESTED_MACRO_BIAS_VERIFIED", "new_fits": 0}), flush=True)
        return
    (reference, h1, c10, best, guard, feature_rows, raw_view, base_config,
     assignments, environment, signature) = preflight(
        config, Path(args.base_config).resolve(), Path(args.inputs).resolve(),
        Path(args.repeat_root).resolve(), h1_oof, c10_oof,
        Path(args.h1_run).resolve(), Path(args.tfidf_run).resolve(),
        Path(args.design_receipt).resolve(), out)
    if args.audit:
        T.write_json(out / "AUDIT_COMPLETE.json", {"status": "AUDIT_COMPLETE", "new_fits": 0})
        return
    biases = {}; bias_receipts = {}; executed = reused = 0
    corrected_c10 = np.full_like(c10["prob"], np.nan)
    corrected_candidate = np.full_like(c10["prob"], np.nan)
    outer_coverage = np.zeros(len(reference["ids"]), dtype=np.int8)
    design = T.read_json(Path(args.design_receipt))
    for outer in config["outer_folds"]:
        outer_indices = assignments[f"outer_{outer}_indices"]
        inner_assignment = assignments[f"outer_{outer}_inner_folds"]
        splits = validate_inner_assignment(reference, outer_indices, inner_assignment,
                                           outer, design["outer"][str(outer)])
        inner_probability = np.full((len(outer_indices), 26), np.nan)
        inner_coverage = np.zeros(len(outer_indices), dtype=np.int8)
        outer_position = {int(global_index): position for position, global_index in enumerate(outer_indices)}
        inner_reviews = []
        for split in splits:
            cache = prepare_inner(config, base_config, feature_rows, raw_view,
                                  reference, outer, split, out)
            probability, was_reused = train_inner(config, reference, cache,
                                                  outer, split["inner"], out)
            positions = np.asarray([outer_position[int(index)] for index in split["valid_indices"]])
            inner_probability[positions] = probability; inner_coverage[positions] += 1
            executed += int(not was_reused); reused += int(was_reused)
            inner_reviews.append(T.read_json(out / "inner_models" / f"outer_{outer}" /
                                             f"inner_{split['inner']}" / "review.json"))
            del cache, probability
            gc.collect()
        if not np.all(inner_coverage == 1) or not np.isfinite(inner_probability).all():
            raise ValueError(f"Nested macro-bias inner OOF incomplete: outer{outer}")
        bias, optimizer_receipt = fit_c10_macro_bias(
            inner_probability, reference["y"][outer_indices], config["bias_optimizer"])
        biases[str(outer)] = bias.tolist()
        outer_valid = reference["folds"] == outer
        corrected = apply_log_bias(c10["prob"][outer_valid], bias, config["bias_optimizer"]["epsilon"])
        corrected_c10[outer_valid] = corrected
        pooled = E.guarded_geometric_pool(h1["prob"][outer_valid], corrected,
                                          guard[outer_valid], 1e-300)
        corrected_candidate[outer_valid] = pooled
        outer_coverage[outer_valid] += 1
        T.save_npz(out / f"outer_{outer}_inner_oof.npz", ids=reference["ids"][outer_indices],
                   y=reference["y"][outer_indices], groups=reference["groups"][outer_indices],
                   inner_folds=inner_assignment, prob=inner_probability)
        bias_receipts[str(outer)] = {
            "outer_train_ids_sha256": T.objsha(reference["ids"][outer_indices].tolist()),
            "inner_oof_sha256": T.sha(out / f"outer_{outer}_inner_oof.npz"),
            "optimizer": optimizer_receipt, "inner_models": inner_reviews,
            "outer_valid_labels_passed_to_optimizer": False,
        }
    if executed > 15 or not np.all(outer_coverage == 1):
        raise ValueError("Nested macro-bias fit budget or outer coverage failure")
    F.validate_oof_probability(corrected_c10, len(reference["ids"]), 26)
    F.validate_oof_probability(corrected_candidate, len(reference["ids"]), 26)
    report(config, reference, best, corrected_c10, corrected_candidate,
           bias_receipts, out, executed, reused)
    models = [f"inner_models/outer_{outer}/inner_{inner}"
              for outer in range(5) for inner in range(3)]
    transforms = [f"inner_transforms/outer_{outer}/inner_{inner}"
                  for outer in range(5) for inner in range(3)]
    T.write_json(out / "NESTED_MACRO_BIAS_COMPLETE.json", {
        "signature": signature, "files": {name: T.sha(out / name) for name in expected_root_files()},
        "model_receipts": models, "transform_receipts": transforms,
        "outer_inner_oof_files": {str(outer): T.sha(out / f"outer_{outer}_inner_oof.npz")
                                  for outer in range(5)},
        "new_cpu_fits": executed, "fits_reused": reused})
    if not completed_run(out, signature):
        raise ValueError("Nested macro-bias final completion verification failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--base-config", required=True)
    parser.add_argument("--inputs", required=True); parser.add_argument("--repeat-root", required=True)
    parser.add_argument("--h1-oof", required=True); parser.add_argument("--c10-oof", required=True)
    parser.add_argument("--h1-run", required=True); parser.add_argument("--tfidf-run", required=True)
    parser.add_argument("--design-receipt", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        T.write_json(Path(args.out) / "FAILED.json", {"error": str(exc),
                     "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
