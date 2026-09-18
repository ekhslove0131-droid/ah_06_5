"""Train-only sparse TF-IDF plus numeric multinomial logistic development cycle."""
from __future__ import annotations

import argparse
import gc
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import goal055_tree_budget as T
import goal055_class_weight as W
import goal055_raw_annotation as R


def symbolic_indices(names):
    return [index for index, name in enumerate(names) if not name.startswith("global|")]


def verify_encoder_contract(spec: dict, prepared_review: dict, train_ids,
                            representation: dict) -> None:
    if spec.get("blocks") != representation.get("blocks"):
        raise ValueError("Fold encoder block contract changed")
    if spec.get("min_support") != representation.get("min_group_support") or spec.get("min_support") != 3:
        raise ValueError("Fold encoder min_support must remain 3")
    if prepared_review.get("train_ids_sha256") != T.objsha(list(train_ids)):
        raise ValueError("Fold encoder training ID receipt mismatch")
    names = spec.get("names", [])
    if not names or len(names) != len(set(names)):
        raise ValueError("Fold encoder feature order is invalid")
    allowed = set(spec.get("fine_allowed", []))
    fine_prefixes = ("exact|", "site|", "gene_change|")
    if any(name.startswith(fine_prefixes) and name not in allowed for name in names):
        raise ValueError("Fold encoder contains an unfitted fine feature")


def fit_feature_blocks(train_symbolic, valid_symbolic, train_numeric, valid_numeric):
    train_symbolic = train_symbolic.astype(np.float64).tocsr()
    valid_symbolic = valid_symbolic.astype(np.float64).tocsr()
    train_numeric = np.asarray(train_numeric, dtype=np.float64)
    valid_numeric = np.asarray(valid_numeric, dtype=np.float64)
    if train_symbolic.shape[1] != valid_symbolic.shape[1]:
        raise ValueError("TF-IDF train/held-out columns differ")
    if train_numeric.ndim != 2 or valid_numeric.ndim != 2 or train_numeric.shape[1] != valid_numeric.shape[1]:
        raise ValueError("Numeric train/held-out columns differ")
    if train_numeric.shape[1] == 0 or not np.isfinite(train_numeric).all() or (train_numeric < 0).any():
        raise ValueError("Train numeric inputs must be finite and nonnegative before log1p")
    if not np.isfinite(valid_numeric).all() or (valid_numeric < 0).any():
        raise ValueError("Held-out numeric inputs must be finite and nonnegative before log1p")
    transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    transformed_train = transformer.fit_transform(train_symbolic).astype(np.float64)
    transformed_valid = transformer.transform(valid_symbolic).astype(np.float64)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_train = scaler.fit_transform(np.log1p(train_numeric))
    scaled_valid = scaler.transform(np.log1p(valid_numeric))
    block_weight = 1.0 / np.sqrt(train_numeric.shape[1])
    scaled_train *= block_weight
    scaled_valid *= block_weight
    combined_train = sparse.hstack(
        [transformed_train, sparse.csr_matrix(scaled_train)], format="csr", dtype=np.float64)
    combined_valid = sparse.hstack(
        [transformed_valid, sparse.csr_matrix(scaled_valid)], format="csr", dtype=np.float64)
    return {
        "train": combined_train, "valid": combined_valid,
        "idf": np.asarray(transformer.idf_, dtype=np.float64),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "numeric_block_weight": block_weight,
    }


def require_convergence(model, captured_warnings, max_iter: int) -> None:
    warning_seen = any(issubclass(item.category, ConvergenceWarning) for item in captured_warnings)
    maxed = np.any(np.asarray(model.n_iter_) >= max_iter)
    if warning_seen or maxed:
        raise ValueError("Logistic regression did not converge within the fixed max_iter")


def validate_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-tfidf-logistic-v1":
        raise ValueError("Wrong TF-IDF logistic version")
    if config.get("seed") != 43 or config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("TF-IDF logistic is restricted to five seed43 folds")
    if config.get("fit_budget") != 10 or config.get("c_values") != [1.0, 10.0]:
        raise ValueError("TF-IDF logistic requires ten fixed C1/C10 fits")
    if config.get("weight_alpha") != 0.5 or config.get("classifier_device") != "CPU":
        raise ValueError("TF-IDF logistic requires alpha0.5 and explicit CPU")
    if config.get("threads") != 4 or config.get("numeric_features") != 41:
        raise ValueError("TF-IDF logistic thread/numeric contract changed")
    if config.get("numeric_block_weight") != "1/sqrt(41)":
        raise ValueError("TF-IDF logistic numeric block weight changed")
    expected_symbolic = {"norm": "l2", "use_idf": True,
                         "smooth_idf": True, "sublinear_tf": True}
    if config.get("symbolic") != expected_symbolic:
        raise ValueError("TF-IDF recipe changed")
    expected_model = {"solver": "lbfgs", "penalty": "l2", "max_iter": 2000,
                      "tol": 0.00001, "fit_intercept": True}
    if config.get("model") != expected_model:
        raise ValueError("Logistic fixed recipe changed")
    if any(config.get(key) is not False for key in ("test_read", "server_score_used", "submission")):
        raise ValueError("Forbidden data flow or external action enabled")


def cpu_environment(base_config: dict, threads: int) -> dict:
    versions = {name: importlib.metadata.version(name) for name in base_config["packages"]}
    if versions != base_config["packages"]:
        raise ValueError("CPU experiment package versions differ from frozen baseline")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "classifier_device": "CPU", "gpu_required": False,
        "thread_limit": threads, "threadpools_before_limit": threadpool_info(),
    }


def run_signature(config: dict, baseline_oof: Path) -> str:
    return T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                     "baseline_oof_sha256": T.sha(baseline_oof)})


def completed_run(out: Path, signature: str) -> bool:
    marker = Path(out) / "TFIDF_LOGISTIC_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker)
    if receipt.get("contract_signature") != signature:
        raise ValueError("Completed TF-IDF logistic run belongs to another configuration")
    files = receipt.get("files")
    expected_models = [f"models/C_{c}/fold_{fold}"
                       for fold in range(5) for c in ("1p0", "10p0")]
    expected_transforms = [f"transforms/fold_{fold}" for fold in range(5)]
    if (not isinstance(files, dict) or not files or
            receipt.get("model_receipts") != expected_models or
            receipt.get("transform_receipts") != expected_transforms):
        raise ValueError("Malformed TF-IDF logistic completion receipt")
    for name, digest in files.items():
        path = Path(out) / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"TF-IDF logistic completion integrity failure: {path}")
    for relative in receipt.get("model_receipts", []) + receipt.get("transform_receipts", []):
        T.verify_complete_receipt(Path(out) / relative)
    return True


def validate_oof_probability(probability, rows: int, classes: int) -> None:
    probability = np.asarray(probability)
    if probability.shape != (rows, classes):
        raise ValueError("OOF probability shape mismatch")
    if (not np.isfinite(probability).all() or (probability < 0).any() or
            not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("OOF probability values are invalid")


def sparse_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def prepare_fold(config: dict, base_config: dict, repeat_root: Path,
                 reference: dict, raw_view, fold: int, out: Path):
    import cbj_lab as B
    import cbj_summary_view as S

    prepared = repeat_root / "seed_43" / "prepared" / f"fold_{fold}"
    expected_marker = config["frozen_fold_receipts"][str(fold)]["prepared"]
    if T.sha(prepared / "COMPLETE.json") != expected_marker:
        raise ValueError(f"Fold {fold} frozen prepared receipt changed")
    T.verify_complete_receipt(prepared)
    matrix_train = sparse.load_npz(prepared / "train.npz").tocsr()
    matrix_valid = sparse.load_npz(prepared / "valid.npz").tocsr()
    spec = B.read_gzip(prepared / "encoder.json.gz")
    review = T.read_json(prepared / "review.json")
    with np.load(prepared / "identity.npz", allow_pickle=False) as identity:
        train_indices = identity["train_indices"]
        valid_indices = identity["valid_indices"]
    T.validate_identity(reference["folds"], reference["groups"], fold,
                        train_indices, valid_indices)
    verify_encoder_contract(spec, review, reference["ids"][train_indices].tolist(),
                            base_config["representation"])
    names = spec["names"]
    if matrix_train.shape[1] != len(names) or matrix_valid.shape[1] != len(names):
        raise ValueError(f"Fold {fold} prepared feature order mismatch")
    global_original = [name for name in names if name.startswith("global|")]
    if global_original != ["global|missing_gene_count", "global|mutation_gene_count",
                           "global|observed_gene_count"]:
        raise ValueError(f"Fold {fold} original global feature contract changed")
    take = symbolic_indices(names)
    symbolic_names = [names[index] for index in take]
    if len(take) != len(names) - 3:
        raise ValueError(f"Fold {fold} symbolic selection did not exclude exactly global block")
    symbolic_train = matrix_train[:, take]
    symbolic_valid = matrix_valid[:, take]
    global_train, global_names, _, _ = S.build_summary_views(
        matrix_train, names, {"placeholder": {"__not_a_gene__"}}, len(base_config["source_genes"]))
    global_valid, valid_global_names, _, _ = S.build_summary_views(
        matrix_valid, names, {"placeholder": {"__not_a_gene__"}}, len(base_config["source_genes"]))
    if global_names != valid_global_names or len(global_names) != 29:
        raise ValueError(f"Fold {fold} global29 order changed")
    numeric_train = np.hstack([global_train.toarray(), raw_view[train_indices].toarray()]).astype(np.float64)
    numeric_valid = np.hstack([global_valid.toarray(), raw_view[valid_indices].toarray()]).astype(np.float64)
    numeric_names = global_names + list(R.RAW_NAMES)
    if len(numeric_names) != config["numeric_features"]:
        raise ValueError(f"Fold {fold} numeric41 order changed")
    transformed = fit_feature_blocks(symbolic_train, symbolic_valid, numeric_train, numeric_valid)
    feature_order = symbolic_names + numeric_names
    if transformed["train"].dtype != np.float64 or transformed["valid"].dtype != np.float64:
        raise ValueError("TF-IDF logistic matrices must remain sparse float64")
    signature = T.objsha({
        "config_preprocessing": {"symbolic": config["symbolic"],
                                 "numeric_block_weight": config["numeric_block_weight"]},
        "fold": fold, "prepared_receipt": expected_marker,
        "train_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
        "feature_order_sha256": T.objsha(feature_order),
    })
    job = Path(out) / "transforms" / f"fold_{fold}"
    if (job / "COMPLETE.json").is_file():
        T.verify_complete_receipt(job, signature)
        with np.load(job / "preprocessor.npz", allow_pickle=False) as stored:
            checks = {
                "idf": transformed["idf"], "scaler_mean": transformed["scaler_mean"],
                "scaler_scale": transformed["scaler_scale"], "symbolic_indices": np.asarray(take),
                "train_indices": train_indices, "valid_indices": valid_indices,
            }
            for key, value in checks.items():
                if not np.array_equal(stored[key], value):
                    raise ValueError(f"Fold {fold} persisted preprocessor changed: {key}")
    else:
        if job.exists() and any(job.iterdir()):
            raise ValueError(f"Incomplete transform artifact exists: {job}")
        job.mkdir(parents=True, exist_ok=True)
        T.save_npz(job / "preprocessor.npz", idf=transformed["idf"],
                   scaler_mean=transformed["scaler_mean"], scaler_scale=transformed["scaler_scale"],
                   symbolic_indices=np.asarray(take, dtype=np.int32),
                   train_indices=train_indices, valid_indices=valid_indices)
        T.write_json(job / "feature_order.json", {
            "feature_order": feature_order, "symbolic_features": symbolic_names,
            "numeric_features": numeric_names,
        })
        T.write_json(job / "review.json", {
            "fold": fold, "prepared_receipt": expected_marker,
            "train_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
            "valid_ids_sha256": T.objsha(reference["ids"][valid_indices].tolist()),
            "encoder_min_support": spec["min_support"],
            "symbolic_features": len(symbolic_names), "numeric_features": len(numeric_names),
            "feature_order_sha256": T.objsha(feature_order),
            "idf_sha256": T.objsha(transformed["idf"].tolist()),
            "scaler_mean_sha256": T.objsha(transformed["scaler_mean"].tolist()),
            "scaler_scale_sha256": T.objsha(transformed["scaler_scale"].tolist()),
            "tfidf_fit_scope": "outer-fold training matrix only",
            "scaler_fit_scope": "outer-fold training numeric41 only",
            "heldout_vocabulary_or_frequency_used": False, "test_read": False,
        })
        files = {name: T.sha(job / name) for name in
                 ("preprocessor.npz", "feature_order.json", "review.json")}
        T.write_json(job / "COMPLETE.json", {"signature": signature, "files": files})
        T.verify_complete_receipt(job, signature)
    coefficient_bytes = len(reference["class_order"]) * transformed["train"].shape[1] * 8
    resource = {
        "fold": fold, "prepared_train_shape": list(matrix_train.shape),
        "prepared_train_nnz": int(matrix_train.nnz),
        "symbolic_features": len(symbolic_names), "numeric_features": len(numeric_names),
        "combined_train_shape": list(transformed["train"].shape),
        "combined_valid_shape": list(transformed["valid"].shape),
        "combined_train_nnz": int(transformed["train"].nnz),
        "combined_train_bytes": sparse_bytes(transformed["train"]),
        "combined_valid_bytes": sparse_bytes(transformed["valid"]),
        "coefficient_bytes_per_model": coefficient_bytes,
        "symbolic_densified": False,
    }
    del matrix_train, matrix_valid, symbolic_train, symbolic_valid, numeric_train, numeric_valid
    gc.collect()
    return {
        **transformed, "train_indices": train_indices, "valid_indices": valid_indices,
        "feature_order": feature_order,
        "transform_receipt_sha256": T.sha(job / "COMPLETE.json"), "resource": resource,
    }


def completed_fit(job: Path, signature: str):
    if not (Path(job) / "COMPLETE.json").is_file():
        return None
    T.verify_complete_receipt(job, signature)
    with np.load(Path(job) / "valid_prob.npz", allow_pickle=False) as stored:
        return np.asarray(stored["prob"], dtype=np.float64)


def train_one(config: dict, reference: dict, cache: dict, c_value: float,
              fold: int, out: Path, environment: dict):
    train_indices = cache["train_indices"]
    valid_indices = cache["valid_indices"]
    weights, weight_receipt = W.fold_local_sample_weights(
        reference["y"][train_indices], len(reference["class_order"]), config["weight_alpha"])
    signature = T.objsha({
        "config": T.objsha(config), "fold": fold, "C": c_value,
        "transform_receipt_sha256": cache["transform_receipt_sha256"],
        "fit_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
        "weights_sha256": T.objsha(weights.tolist()),
    })
    c_name = str(c_value).replace(".", "p")
    model_root = Path(out) / "models" / f"C_{c_name}"
    job = model_root / f"fold_{fold}"
    reused = completed_fit(job, signature)
    if reused is not None:
        T.emit(out, "TFIDF_LOGISTIC_FOLD_REUSED", fold=fold, C=c_value)
        return reused, True
    model_root.mkdir(parents=True, exist_ok=True)
    partial = model_root / f".fold_{fold}.{os.getpid()}.partial"
    partial.mkdir()
    params = dict(config["model"], C=c_value)
    model = LogisticRegression(**params)
    T.emit(out, "TFIDF_LOGISTIC_CPU_FIT_START", fold=fold, C=c_value,
           rows=len(train_indices), features=cache["train"].shape[1], threads=config["threads"])
    started = time.perf_counter()
    with threadpool_limits(limits=config["threads"]):
        effective_threadpools = threadpool_info()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(cache["train"], reference["y"][train_indices], sample_weight=weights)
    fit_seconds = time.perf_counter() - started
    require_convergence(model, captured, config["model"]["max_iter"])
    if model.classes_.tolist() != list(range(len(reference["class_order"]))):
        raise ValueError("TF-IDF logistic class order changed")
    train_probability = np.asarray(model.predict_proba(cache["train"]), dtype=np.float64)
    probability = np.asarray(model.predict_proba(cache["valid"]), dtype=np.float64)
    joblib.dump(model, partial / "model.joblib", compress=3)
    T.save_npz(partial / "coefficients.npz", coef=model.coef_, intercept=model.intercept_,
               classes=model.classes_, n_iter=model.n_iter_)
    restored = joblib.load(partial / "model.joblib")
    probe = np.asarray(restored.predict_proba(cache["valid"][:min(16, len(valid_indices))]),
                       dtype=np.float64)
    reload_max_abs = float(np.max(np.abs(probe - probability[:len(probe)])))
    if reload_max_abs >= config["reload_tolerance"]:
        raise ValueError("TF-IDF logistic reload prediction mismatch")
    train_metrics, _ = T.metric_summary(T.make_fold_reference(reference, train_indices), train_probability)
    valid_metrics, _ = T.metric_summary(T.make_fold_reference(reference, valid_indices), probability)
    T.save_npz(partial / "valid_prob.npz", prob=probability)
    T.write_json(partial / "review.json", {
        "fold": fold, "C": c_value, "fit_seconds": fit_seconds,
        "requested_parameters": params, "n_iter": model.n_iter_.tolist(),
        "convergence_status": "CONVERGED", "warnings": [str(item.message) for item in captured],
        "classes": model.classes_.tolist(), "coef_shape": list(model.coef_.shape),
        "feature_order_sha256": T.objsha(cache["feature_order"]),
        "transform_receipt_sha256": cache["transform_receipt_sha256"],
        "fit_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
        "weight_receipt": weight_receipt, "classifier_device": "CPU",
        "effective_threadpools": effective_threadpools,
        "train_metrics": train_metrics, "valid_metrics": valid_metrics,
        "reload_max_abs": reload_max_abs, "test_read": False,
        "server_score_used": False, "submission_created": False,
    })
    files = {name: T.sha(partial / name) for name in
             ("model.joblib", "coefficients.npz", "valid_prob.npz", "review.json")}
    T.write_json(partial / "COMPLETE.json", {"signature": signature, "files": files})
    T.verify_complete_receipt(partial, signature)
    if job.exists():
        raise ValueError(f"TF-IDF logistic fold destination appeared during fit: {job}")
    partial.replace(job)
    T.emit(out, "TFIDF_LOGISTIC_CPU_FIT_COMPLETE", fold=fold, C=c_value,
           fit_seconds=fit_seconds, n_iter=model.n_iter_.tolist(),
           valid_macro_f1=valid_metrics["macro_f1"])
    del model, restored
    gc.collect()
    return probability, False


def preflight(config: dict, base_config_path: Path, inputs: Path,
              repeat_root: Path, baseline_oof: Path, out: Path):
    import lab_metrics as L

    validate_config(config)
    helpers = {
        "tree_helper_sha256": Path(T.__file__), "weight_helper_sha256": Path(W.__file__),
        "raw_helper_sha256": Path(R.__file__),
    }
    for key, path in helpers.items():
        if T.sha(path) != config.get(key):
            raise ValueError(f"Pinned TF-IDF logistic helper mismatch: {key}")
    T.verify_named_hashes(T.PROJECT_ROOT, config["frozen_source_hashes"], "frozen source")
    if T.sha(base_config_path) != config["base_config_sha256"]:
        raise ValueError("TF-IDF logistic base config hash mismatch")
    T.verify_named_hashes(inputs, config["input_hashes"], "training input")
    T.verify_named_hashes(repeat_root, config["repeat_root_hashes"], "repeat artifact")
    if T.sha(baseline_oof) != config["baseline_oof_sha256"]:
        raise ValueError("TF-IDF logistic H1 baseline OOF hash mismatch")
    base_config = T.read_json(base_config_path)
    reference = L.load_oof(repeat_root / "seed_43" / "mrmr_global_oof.npz")
    baseline = L.load_oof(baseline_oof)
    for key in ("ids", "y", "groups", "folds", "class_order"):
        if not np.array_equal(reference[key], baseline[key]):
            raise ValueError(f"TF-IDF logistic baseline {key} mismatch")
    raw = pd.read_csv(inputs / base_config["inputs"]["train"]["filename"], dtype=str)
    if raw.columns.tolist() != ["ID", "SUBCLASS"] + base_config["source_genes"]:
        raise ValueError("TF-IDF logistic raw schema/gene order mismatch")
    if raw.ID.tolist() != reference["ids"].tolist():
        raise ValueError("TF-IDF logistic raw ID order mismatch")
    if raw.SUBCLASS.tolist() != reference["class_order"][reference["y"]].tolist():
        raise ValueError("TF-IDF logistic raw target identity mismatch")
    raw_view, raw_names, raw_stats = R.build_raw_annotation_view(
        raw[base_config["source_genes"]], base_config["source_genes"])
    del raw
    gc.collect()
    if raw_names != list(R.RAW_NAMES):
        raise ValueError("TF-IDF logistic raw12 order mismatch")
    R.validate_raw_stats(raw_stats, config["expected_raw_stats"])

    signature = run_signature(config, baseline_oof)
    out.mkdir(parents=True, exist_ok=True)
    contract = out / "CONTRACT.json"
    if contract.exists():
        if T.read_json(contract).get("signature") != signature:
            raise ValueError("TF-IDF logistic output belongs to another configuration")
    else:
        unexpected = sorted(path.name for path in out.iterdir()
                            if path.name not in {"launch.json", "execution.log"})
        if unexpected:
            raise ValueError("TF-IDF logistic output contains unrelated artifacts")
        T.write_json(contract, {
            "signature": signature, "config": config, "runner_sha256": T.sha(Path(__file__)),
            "baseline_oof": str(baseline_oof),
            "scope": "full fold-fitted MCMP symbolic TF-IDF plus train-scaled global29/raw12 CPU logistic",
        })
    environment = cpu_environment(base_config, config["threads"])
    T.write_json(out / "environment.json", environment)
    caches = {fold: prepare_fold(config, base_config, repeat_root, reference, raw_view, fold, out)
              for fold in config["folds"]}
    resources = {str(fold): caches[fold]["resource"] for fold in config["folds"]}
    total_cached_bytes = sum(item["combined_train_bytes"] + item["combined_valid_bytes"]
                             for item in resources.values())
    max_coefficient_bytes = max(item["coefficient_bytes_per_model"] for item in resources.values())
    available = __import__("psutil").virtual_memory().available
    estimated_peak = total_cached_bytes + max_coefficient_bytes * 30 + 2**30
    if estimated_peak >= min(base_config["runtime"]["max_ram_gib"] * 2**30, available * 0.8):
        raise ValueError("CPU TF-IDF logistic memory admission failed")
    admission = {
        "rows": len(reference["ids"]), "classes": len(reference["class_order"]),
        "canonical_groups": len(np.unique(reference["groups"])),
        "raw_stats": raw_stats, "fold_resources": resources,
        "total_cached_sparse_bytes": total_cached_bytes,
        "max_coefficient_bytes": max_coefficient_bytes,
        "estimated_peak_bytes": estimated_peak, "available_ram_bytes": available,
        "classifier_device": "CPU", "thread_limit": config["threads"],
        "gpu_required": False, "fit_budget": 10,
        "heldout_vocabulary_or_frequency_used": False, "test_read": False,
    }
    T.write_json(out / "ADMISSION.json", admission)
    T.emit(out, "TFIDF_LOGISTIC_PREFLIGHT_COMPLETE", rows=admission["rows"],
           fit_budget=10, classifier_device="CPU", threads=config["threads"],
           estimated_peak_bytes=estimated_peak)
    return reference, baseline, caches, environment, signature


def report(config: dict, reference: dict, baseline: dict,
           predictions: dict[float, np.ndarray], out: Path, executed: int, reused: int):
    import lab_metrics as L

    endpoints = {"h1_xgb_alpha0p5": baseline["prob"],
                 **{f"tfidf_logistic_C{str(c).replace('.', 'p')}": probability
                    for c, probability in predictions.items()}}
    scores = {name: L.metrics(reference, value) for name, value in endpoints.items()}
    rows = []
    for name, score in scores.items():
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])}})
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        L.save_npz(out / f"{name}_oof.npz", **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=endpoints[name])
    pd.DataFrame(rows).to_csv(out / "tfidf_logistic_scoreboard.csv", index=False)
    comparisons = {}
    for name in endpoints:
        if name == "h1_xgb_alpha0p5":
            continue
        comparisons[f"{name}_vs_h1"] = {
            "delta_macro_f1": scores[name]["macro_f1"] - scores["h1_xgb_alpha0p5"]["macro_f1"],
            **L.changes(reference["y"], endpoints[name], baseline["prob"]),
            "bootstrap": L.bootstrap(reference, endpoints[name], baseline["prob"],
                                     config["bootstrap_resamples"], 43),
        }
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_TFIDF_LOGISTIC_COMPLETE", "seed": 43,
        "scores": {name: {key: value for key, value in score.items() if key != "per_class"}
                   for name, score in scores.items()},
        "comparisons": comparisons, "fits_executed": executed, "fits_reused": reused,
        "fit_budget": 10, "classifier_device": "CPU", "thread_limit": config["threads"],
        "automatic_promotion": False, "test_read": False,
        "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "TFIDF_LOGISTIC_REVIEW.json", review)
    return review


def run(args):
    config = T.read_json(Path(args.config))
    out = Path(args.out).resolve()
    signature = run_signature(config, Path(args.baseline_oof).resolve())
    if completed_run(out, signature):
        print(json.dumps({"event": "COMPLETED_TFIDF_LOGISTIC_VERIFIED", "new_fits": 0}), flush=True)
        return
    reference, baseline, caches, environment, signature = preflight(
        config, Path(args.base_config).resolve(), Path(args.inputs).resolve(),
        Path(args.repeat_root).resolve(), Path(args.baseline_oof).resolve(), out)
    if args.audit:
        T.write_json(out / "AUDIT_COMPLETE.json", {"status": "AUDIT_COMPLETE", "new_fits": 0})
        return
    predictions = {c: np.full_like(reference["prob"], np.nan) for c in config["c_values"]}
    coverage = {c: np.zeros(len(reference["ids"]), dtype=np.int8) for c in config["c_values"]}
    executed = reused = 0
    for fold in config["folds"]:
        for c_value in config["c_values"]:
            probability, was_reused = train_one(
                config, reference, caches[fold], c_value, fold, out, environment)
            indices = caches[fold]["valid_indices"]
            predictions[c_value][indices] = probability
            coverage[c_value][indices] += 1
            executed += int(not was_reused)
            reused += int(was_reused)
    if executed > 10 or any(not np.all(value == 1) for value in coverage.values()):
        raise ValueError("TF-IDF logistic fit budget or OOF coverage failure")
    for probability in predictions.values():
        validate_oof_probability(probability, len(reference["ids"]), len(reference["class_order"]))
    review = report(config, reference, baseline, predictions, out, executed, reused)
    files = ["TFIDF_LOGISTIC_REVIEW.json", "tfidf_logistic_scoreboard.csv", "ADMISSION.json",
             "h1_xgb_alpha0p5_oof.npz", "h1_xgb_alpha0p5_per_class.csv", "environment.json"]
    for c_value in config["c_values"]:
        name = f"tfidf_logistic_C{str(c_value).replace('.', 'p')}"
        files += [f"{name}_oof.npz", f"{name}_per_class.csv"]
    model_receipts = [f"models/C_{str(c).replace('.', 'p')}/fold_{fold}"
                      for fold in config["folds"] for c in config["c_values"]]
    transform_receipts = [f"transforms/fold_{fold}" for fold in config["folds"]]
    T.write_json(out / "TFIDF_LOGISTIC_COMPLETE.json", {
        "status": review["status"], "contract_signature": signature,
        "files": {name: T.sha(out / name) for name in files},
        "model_receipts": model_receipts, "transform_receipts": transform_receipts,
        "new_cpu_fits": executed,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--repeat-root", required=True)
    parser.add_argument("--baseline-oof", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        T.write_json(Path(args.out) / "FAILED.json", {
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
