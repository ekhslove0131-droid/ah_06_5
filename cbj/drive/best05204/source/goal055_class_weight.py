"""Fold-train-only class-weight experiment driver for the CBJ goal-0.55 lab."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np

import goal055_tree_budget as T


def fold_local_sample_weights(y_train, class_count: int, alpha: float):
    y_train = np.asarray(y_train)
    if alpha not in (0.5, 1.0):
        raise ValueError("Only predeclared alpha 0.5 or 1.0 is allowed")
    if y_train.ndim != 1 or len(y_train) == 0 or y_train.dtype.kind not in "iu":
        raise ValueError("Invalid fold-training labels")
    counts = np.bincount(y_train, minlength=class_count)
    if len(counts) != class_count or np.any(counts == 0):
        raise ValueError("Fold-training labels contain a missing class")
    class_weights = np.power(len(y_train) / counts.astype(np.float64), alpha)
    weights = class_weights[y_train]
    weights /= weights.mean()
    receipt = {
        "alpha": alpha, "fit_rows": len(y_train), "class_counts": counts.tolist(),
        "class_weights_after_mean_normalization": (class_weights / class_weights[y_train].mean()).tolist(),
        "sample_weight_mean": float(weights.mean()),
        "scope": "outer-fold training labels only",
    }
    return weights, receipt


def validate_weight_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-class-weight-v1":
        raise ValueError("Wrong class-weight experiment version")
    if config.get("seed") != 43 or config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("Class-weight development screen is restricted to five seed43 folds")
    if config.get("alphas") != [0.5, 1.0] or config.get("fit_budget") != 10:
        raise ValueError("Exactly two predeclared alphas and ten fits are required")
    if config.get("condition") != "mrmr_global":
        raise ValueError("Only mrmr_global is allowed")
    trees = config.get("model", {}).get("n_estimators")
    if not isinstance(trees, int) or trees <= 0:
        raise ValueError("An explicit positive n_estimators tree budget is required")
    if config.get("model") != config.get("baseline_model"):
        raise ValueError("Class weights must be the sole variable from baseline_model")
    forbidden_model_keys = {"early_stopping_rounds", "callbacks"} & set(config.get("model", {}))
    if forbidden_model_keys:
        raise ValueError("Early stopping and callbacks are forbidden")
    if any(config.get(key) is not False for key in ("test_read", "server_score_used", "submission")):
        raise ValueError("Forbidden data flow or external action enabled")


def verify_helper_source(config: dict) -> str:
    observed = T.sha(Path(T.__file__))
    if observed != config.get("tree_helper_sha256"):
        raise ValueError("Pinned tree helper source hash mismatch")
    return observed


def weight_contract_signature(config: dict, baseline_oof: Path) -> str:
    return T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                     "baseline_oof_sha256": T.sha(baseline_oof)})


def completed_weight_run(out: Path, signature: str) -> bool:
    marker = Path(out) / "CLASS_WEIGHT_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker)
    if receipt.get("contract_signature") != signature:
        raise ValueError("Completed weighted run belongs to another configuration")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Malformed completed weighted run receipt")
    for name, digest in files.items():
        path = Path(out) / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Completed weighted run integrity failure: {path}")
    return True


def completed_weight_fit(job: Path, signature: str):
    if not (Path(job) / "COMPLETE.json").is_file():
        return None
    T.verify_complete_receipt(job, signature)
    with np.load(Path(job) / "valid_prob.npz", allow_pickle=False) as stored:
        return np.asarray(stored["prob"], dtype=np.float64)


def train_weight_fold(config: dict, base_config: dict, reference: dict, cache: dict,
                      fold: int, alpha: float, out: Path, environment: dict):
    import pandas as pd
    import psutil
    import xgboost as xgb

    y_train = reference["y"][cache["train_indices"]]
    weights, weight_receipt = fold_local_sample_weights(
        y_train, len(reference["class_order"]), alpha)
    signature = T.objsha({
        "config": T.objsha(config), "fold": fold, "alpha": alpha,
        "fit_ids_sha256": T.objsha(reference["ids"][cache["train_indices"]].tolist()),
        "identity_sha256": cache["identity_sha256"], "mask_sha256": cache["mask_sha256"],
        "names_sha256": T.objsha(cache["names"]), "weights_sha256": T.objsha(weights.tolist()),
    })
    alpha_name = str(alpha).replace(".", "p")
    model_root = Path(out) / "models" / f"alpha_{alpha_name}"
    job = model_root / f"fold_{fold}"
    reused = completed_weight_fit(job, signature)
    if reused is not None:
        T.emit(out, "WEIGHT_FOLD_REUSED", fold=fold, alpha=alpha)
        return reused, True
    model_root.mkdir(parents=True, exist_ok=True)
    partial = model_root / f".fold_{fold}.{os.getpid()}.partial"
    partial.mkdir()
    runtime = base_config["runtime"]
    dense_bytes = (cache["train"].shape[0] + cache["valid"].shape[0]) * cache["train"].shape[1] * 4
    estimated = psutil.Process().memory_info().rss + dense_bytes * runtime["dense_multiplier"] + 2**30
    if estimated >= runtime["max_ram_gib"] * 2**30:
        raise ValueError("Insufficient RAM for weighted dense training")
    dense_train = cache["train"].toarray(order="C")
    dense_valid = cache["valid"].toarray(order="C")
    params = dict(config["model"])
    params.update(device=f"cuda:{runtime['gpu_index']}", n_jobs=runtime["threads"],
                  num_class=len(reference["class_order"]))
    model = xgb.XGBClassifier(**params)
    T.emit(out, "WEIGHT_GPU_FIT_START", fold=fold, alpha=alpha,
           trees=params["n_estimators"], features=len(cache["names"]))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(dense_train, y_train, sample_weight=weights,
                  eval_set=[(dense_train, y_train),
                            (dense_valid, reference["y"][cache["valid_indices"]])],
                  verbose=False)
    fit_seconds = time.perf_counter() - started
    booster = model.get_booster()
    effective = json.loads(booster.save_config())
    if not effective["learner"]["generic_param"]["device"].startswith("cuda"):
        raise ValueError("Weighted classifier did not use CUDA")
    if booster.num_boosted_rounds() != params["n_estimators"]:
        raise ValueError("Weighted classifier tree budget changed")
    if model.classes_.tolist() != list(range(len(reference["class_order"]))):
        raise ValueError("Weighted classifier class order changed")
    probability = T.predict_booster(booster, cache["valid"], runtime["predict_batch_rows"])
    evals = model.evals_result()
    learning_rows = [{
        "iteration": index + 1,
        "train_unweighted_mlogloss": float(evals["validation_0"]["mlogloss"][index]),
        "valid_unweighted_mlogloss": float(evals["validation_1"]["mlogloss"][index]),
    } for index in range(params["n_estimators"])]
    pd.DataFrame(learning_rows).to_csv(partial / "learning_curve.csv", index=False)
    T.save_npz(partial / "valid_prob.npz", prob=probability)
    model.save_model(partial / "model.ubj")
    T.write_json(partial / "review.json", {
        "fold": fold, "alpha": alpha, "fit_seconds": fit_seconds,
        "fit_ids_sha256": T.objsha(reference["ids"][cache["train_indices"]].tolist()),
        "weight_receipt": weight_receipt, "validation_weights": None,
        "requested_parameters": params, "effective_parameters": effective,
        "gpu": environment["gpu"], "estimated_ram_bytes": estimated,
        "warnings": [str(item.message) for item in captured],
        "train_only": True, "test_read": False, "server_score_used": False,
    })
    files = {name: T.sha(partial / name) for name in
             ("model.ubj", "valid_prob.npz", "learning_curve.csv", "review.json")}
    T.write_json(partial / "COMPLETE.json", {"signature": signature, "files": files})
    T.verify_complete_receipt(partial, signature)
    if job.exists():
        raise ValueError(f"Weighted fold destination appeared during fit: {job}")
    partial.replace(job)
    T.emit(out, "WEIGHT_GPU_FIT_COMPLETE", fold=fold, alpha=alpha, fit_seconds=fit_seconds)
    del model, booster, dense_train, dense_valid
    gc.collect()
    return probability, False


def load_weight_sources(config: dict, base_config_path: Path, inputs: Path,
                        repeat_root: Path, baseline_oof: Path, out: Path):
    import lab_metrics as L

    validate_weight_config(config)
    verify_helper_source(config)
    T.verify_named_hashes(T.PROJECT_ROOT, config["frozen_source_hashes"], "frozen source")
    if T.sha(base_config_path) != config["base_config_sha256"]:
        raise ValueError("Base config hash mismatch")
    T.verify_named_hashes(inputs, config["input_hashes"], "training input")
    T.verify_named_hashes(repeat_root, config["repeat_root_hashes"], "repeat artifact")
    if T.sha(baseline_oof) != config["baseline_oof_sha256"]:
        raise ValueError("Unweighted baseline OOF hash mismatch")
    base_config = T.read_json(base_config_path)
    seed_reference = L.load_oof(repeat_root / "seed_43" / "mrmr_global_oof.npz")
    baseline = L.load_oof(baseline_oof)
    for key in ("ids", "y", "groups", "folds", "class_order"):
        if not np.array_equal(seed_reference[key], baseline[key]):
            raise ValueError(f"Unweighted baseline {key} mismatch")
    environment = T.environment_receipt(base_config, True)
    caches = {fold: T.build_fold_cache(config, base_config, repeat_root, seed_reference, fold)
              for fold in config["folds"]}
    contract = weight_contract_signature(config, baseline_oof)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "CONTRACT.json").exists():
        if T.read_json(out / "CONTRACT.json").get("signature") != contract:
            raise ValueError("Weighted output directory belongs to another configuration")
    else:
        unexpected = sorted(path.name for path in out.iterdir()
                            if path.name not in {"launch.json", "execution.log"})
        if unexpected:
            raise ValueError("Weighted output directory contains unrelated artifacts")
        T.write_json(out / "CONTRACT.json", {
            "signature": contract, "config": config, "runner_sha256": T.sha(Path(__file__)),
            "baseline_oof": str(baseline_oof), "scope": "seed43 fold-train-only class weights",
        })
    T.write_json(out / "environment.json", environment)
    return base_config, seed_reference, baseline, caches, environment, contract


def report_weights(config: dict, reference: dict, baseline: dict,
                   predictions: dict[float, np.ndarray], out: Path, executed: int, reused: int):
    import pandas as pd
    import lab_metrics as L

    endpoints = {"unweighted": baseline["prob"],
                 **{f"alpha_{str(alpha).replace('.', 'p')}": value
                    for alpha, value in predictions.items()}}
    scores = {name: L.metrics(reference, probability) for name, probability in endpoints.items()}
    rows = []
    for name, score in scores.items():
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])}})
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        L.save_npz(out / f"{name}_oof.npz", **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=endpoints[name])
    pd.DataFrame(rows).to_csv(out / "weight_scoreboard.csv", index=False)
    comparisons = {}
    for name in endpoints:
        if name == "unweighted":
            continue
        comparisons[f"{name}_vs_unweighted"] = {
            "delta_macro_f1": scores[name]["macro_f1"] - scores["unweighted"]["macro_f1"],
            **L.changes(reference["y"], endpoints[name], endpoints["unweighted"]),
            "bootstrap": L.bootstrap(reference, endpoints[name], endpoints["unweighted"],
                                     config["bootstrap_resamples"], 43),
        }
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_CLASS_WEIGHT_COMPLETE",
        "seed": 43, "scores": {name: {key: value for key, value in score.items()
                                        if key != "per_class"} for name, score in scores.items()},
        "comparisons": comparisons, "fits_executed": executed, "fits_reused": reused,
        "fit_budget": 10, "weight_fit_scope": "outer-fold training labels only",
        "validation_weights": None, "automatic_promotion": False,
        "test_read": False, "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "CLASS_WEIGHT_REVIEW.json", review)
    return review


def run(args):
    config = T.read_json(Path(args.config))
    out = Path(args.out).resolve()
    signature = weight_contract_signature(config, Path(args.baseline_oof).resolve())
    if completed_weight_run(out, signature):
        print(json.dumps({"event": "COMPLETED_WEIGHT_RUN_VERIFIED", "new_classifier_fits": 0}), flush=True)
        return
    base_config, reference, baseline, caches, environment, contract = load_weight_sources(
        config, Path(args.base_config).resolve(), Path(args.inputs).resolve(),
        Path(args.repeat_root).resolve(), Path(args.baseline_oof).resolve(), out)
    if args.audit:
        T.write_json(out / "AUDIT_COMPLETE.json", {"status": "AUDIT_COMPLETE", "new_fits": 0})
        return
    predictions = {alpha: np.full_like(reference["prob"], np.nan) for alpha in config["alphas"]}
    coverage = {alpha: np.zeros(len(reference["ids"]), dtype=np.int8) for alpha in config["alphas"]}
    executed = reused = 0
    for alpha in config["alphas"]:
        for fold in config["folds"]:
            probability, was_reused = train_weight_fold(
                config, base_config, reference, caches[fold], fold, alpha, out, environment)
            predictions[alpha][caches[fold]["valid_indices"]] = probability
            coverage[alpha][caches[fold]["valid_indices"]] += 1
            reused += int(was_reused)
            executed += int(not was_reused)
    if executed > config["fit_budget"] or any(not np.all(value == 1) for value in coverage.values()):
        raise ValueError("Weighted fit budget or OOF coverage failure")
    review = report_weights(config, reference, baseline, predictions, out, executed, reused)
    root_files = ["CLASS_WEIGHT_REVIEW.json", "weight_scoreboard.csv", "environment.json"]
    root_files += ["unweighted_oof.npz", "unweighted_per_class.csv"]
    for alpha in config["alphas"]:
        name = f"alpha_{str(alpha).replace('.', 'p')}"
        root_files += [f"{name}_oof.npz", f"{name}_per_class.csv"]
    T.write_json(out / "CLASS_WEIGHT_COMPLETE.json", {
        "status": review["status"], "contract_signature": contract,
        "files": {name: T.sha(out / name) for name in root_files},
        "new_gpu_fits": executed,
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
