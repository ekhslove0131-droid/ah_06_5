"""Train-only seed-43 XGBoost tree-budget diagnostic using frozen E2-C caches."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_complete_receipt(root: Path, expected_signature: str | None = None) -> dict[str, str]:
    root = Path(root)
    marker = root / "COMPLETE.json"
    if not marker.is_file():
        raise ValueError(f"Missing completion receipt: {marker}")
    receipt = read_json(marker)
    if expected_signature is not None and receipt.get("signature") != expected_signature:
        raise ValueError(f"Existing artifact belongs to another configuration: {root}")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Malformed completion receipt: {marker}")
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Cached artifact integrity failure: {path}")
    return dict(files)


def validate_identity(folds, groups, fold: int, train_indices, valid_indices) -> None:
    folds = np.asarray(folds)
    groups = np.asarray(groups)
    train_indices = np.asarray(train_indices)
    valid_indices = np.asarray(valid_indices)
    expected_train = np.flatnonzero(folds != fold)
    expected_valid = np.flatnonzero(folds == fold)
    if not np.array_equal(train_indices, expected_train):
        raise ValueError(f"Fold {fold} training indices are not exact")
    if not np.array_equal(valid_indices, expected_valid):
        raise ValueError(f"Fold {fold} validation indices are not exact")
    if set(groups[train_indices]) & set(groups[valid_indices]):
        raise ValueError(f"Canonical group crosses fold {fold}")


def validate_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-tree-budget-v1":
        raise ValueError("Wrong experiment version")
    if config.get("seed") != 43:
        raise ValueError("This adaptive screen is restricted to seed 43")
    if config.get("folds") != [0, 1, 2, 3, 4] or config.get("fit_budget") != 5:
        raise ValueError("Exactly five seed-43 folds are required")
    if config.get("condition") != "mrmr_global":
        raise ValueError("Only mrmr_global is allowed")
    if config.get("diagnostic_prefixes") != [200, 400, 800]:
        raise ValueError("Diagnostic prefixes must be 200, 400, 800")
    if any(config.get(key) is not False for key in ("test_read", "server_score_used", "submission")):
        raise ValueError("Forbidden data flow or external action enabled")
    model = config.get("model", {})
    baseline = config.get("baseline_model", {})
    if model.get("n_estimators") != 800 or baseline.get("n_estimators") != 200:
        raise ValueError("Tree budgets must be 800 and 200")
    differing = {key for key in set(model) | set(baseline) if model.get(key) != baseline.get(key)}
    if differing != {"n_estimators"}:
        raise ValueError("Experiment may change only n_estimators")


def completed_fit(job: Path, signature: str) -> dict[str, np.ndarray] | None:
    job = Path(job)
    if not (job / "COMPLETE.json").is_file():
        return None
    verify_complete_receipt(job, expected_signature=signature)
    with np.load(job / "prefix_probabilities.npz", allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def completed_run(out: Path, contract_signature: str) -> bool:
    out = Path(out)
    marker = out / "GOAL055_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = read_json(marker)
    if receipt.get("contract_signature") != contract_signature:
        raise ValueError("Completed run belongs to another configuration")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Malformed completed run receipt")
    for name, digest in files.items():
        path = out / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Completed run integrity failure: {path}")
    return True


def objsha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def contract_signature(config: dict, base_config_path: Path) -> str:
    return objsha({"config": config, "runner_sha256": sha(Path(__file__)),
                   "base_config_sha256": sha(base_config_path)})


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    partial.replace(path)


def save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in arrays.items()})
    partial.replace(path)


def emit(out: Path, event: str, **values) -> None:
    row = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "event": event, **values}
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    with (Path(out) / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def verify_named_hashes(root: Path, expected: dict[str, str], label: str) -> dict[str, str]:
    observed = {}
    for name, digest in expected.items():
        path = Path(root) / name
        if not path.is_file():
            raise ValueError(f"Missing {label}: {path}")
        observed[name] = sha(path)
        if observed[name] != digest:
            raise ValueError(f"{label} hash mismatch: {name}")
    return observed


def environment_receipt(base_config: dict, training: bool) -> dict:
    packages = {name: importlib.metadata.version(name) for name in base_config["packages"]}
    if packages != base_config["packages"]:
        raise ValueError("Package versions differ from frozen baseline")
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "training_requested": training, "gpu": None}
    if training:
        import xgboost as xgb
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        rows = []
        for line in text.strip().splitlines():
            index, name, uuid, driver, total, free, util = [part.strip() for part in line.split(",")]
            rows.append({"index": int(index), "name": name, "uuid": uuid, "driver": driver,
                         "total_mib": int(total), "free_mib": int(free), "utilization_percent": int(util)})
        wanted = [row for row in rows if row["index"] == base_config["runtime"]["gpu_index"]]
        if len(wanted) != 1 or wanted[0]["free_mib"] < base_config["runtime"]["minimum_free_mib"]:
            raise ValueError("GPU unavailable or insufficient free VRAM")
        if xgb.build_info().get("USE_CUDA") is not True:
            raise ValueError("XGBoost CUDA build required; no CPU fallback")
        result["gpu"] = wanted[0]
    return result


def fold_signature(config: dict, fold: int, identity_hash: str,
                   names_hash: str, mask_hash: str) -> str:
    return objsha({
        "experiment": objsha(config), "fold": fold, "identity_sha256": identity_hash,
        "names_sha256": names_hash, "mask_sha256": mask_hash,
        "condition": "mrmr_global", "train_only": True,
    })


def predict_booster(booster, matrix, rows_per_batch: int, prefix: int | None = None):
    import xgboost as xgb
    pieces = []
    iteration_range = (0, prefix) if prefix is not None else (0, 0)
    for start in range(0, matrix.shape[0], rows_per_batch):
        dense = matrix[start:start + rows_per_batch].toarray(order="C")
        pieces.append(booster.predict(xgb.DMatrix(dense), iteration_range=iteration_range,
                                      strict_shape=True))
    result = np.concatenate(pieces).astype(np.float64, copy=False)
    if result.ndim != 2 or not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid prefix probability output")
    return result / result.sum(1, keepdims=True)


def metric_summary(reference: dict, probability) -> tuple[dict, list[dict]]:
    import lab_metrics as L
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = L.metrics(reference, probability)
    compact = {key: value for key, value in score.items() if key not in ("per_class", "fold_f1")}
    if not np.isfinite(compact["fold_sd"]):
        compact["fold_sd"] = None
    compact["fold_f1"] = score["fold_f1"]
    return compact, score["per_class"]


def make_fold_reference(reference: dict, indices: np.ndarray) -> dict:
    return {key: reference[key] if key == "class_order" else reference[key][indices]
            for key in ("ids", "y", "groups", "folds", "class_order")}


def build_fold_cache(config: dict, base_config: dict, repeat_root: Path,
                     reference: dict, fold: int):
    from scipy import sparse
    import cbj_lab as B
    import cbj_summary_view as S

    expected = config["frozen_fold_receipts"][str(fold)]
    prepared = repeat_root / "seed_43" / "prepared" / f"fold_{fold}"
    selection = repeat_root / "seed_43" / "selection" / f"fold_{fold}"
    baseline_job = repeat_root / "seed_43" / "models" / "mrmr_global" / f"fold_{fold}"
    for label, directory in (("prepared", prepared), ("selection", selection),
                             ("baseline_model", baseline_job)):
        marker_hash = sha(directory / "COMPLETE.json")
        if marker_hash != expected[label]:
            raise ValueError(f"Frozen fold {fold} {label} receipt changed")
        verify_complete_receipt(directory)

    train_matrix = sparse.load_npz(prepared / "train.npz").tocsr()
    valid_matrix = sparse.load_npz(prepared / "valid.npz").tocsr()
    names = B.read_gzip(prepared / "encoder.json.gz")["names"]
    if train_matrix.shape[1] != len(names) or valid_matrix.shape[1] != len(names):
        raise ValueError(f"Fold {fold} encoder/matrix column mismatch")
    with np.load(prepared / "identity.npz", allow_pickle=False) as identity:
        train_indices = identity["train_indices"]
        valid_indices = identity["valid_indices"]
    validate_identity(reference["folds"], reference["groups"], fold, train_indices, valid_indices)

    with np.load(selection / "masks.npz", allow_pickle=False) as masks:
        take = np.asarray(masks["mrmr"], dtype=np.int64)
    if take.ndim != 1 or len(take) == 0 or len(np.unique(take)) != len(take):
        raise ValueError(f"Fold {fold} invalid mRMR mask")
    if not np.array_equal(take, np.sort(take)) or take.min() < 0 or take.max() >= len(names):
        raise ValueError(f"Fold {fold} mRMR mask order/range mismatch")

    global_train, global_names, _, _ = S.build_summary_views(
        train_matrix, names, {"placeholder": {"__not_a_gene__"}}, len(base_config["source_genes"]))
    global_valid, valid_global_names, _, _ = S.build_summary_views(
        valid_matrix, names, {"placeholder": {"__not_a_gene__"}}, len(base_config["source_genes"]))
    if global_names != valid_global_names or len(global_names) != 29:
        raise ValueError(f"Fold {fold} global summary mismatch")
    combined_train = sparse.hstack([train_matrix[:, take], global_train], format="csr", dtype=np.float32)
    combined_valid = sparse.hstack([valid_matrix[:, take], global_valid], format="csr", dtype=np.float32)
    combined_names = [names[index] for index in take] + global_names

    with np.load(baseline_job / "columns.npz", allow_pickle=False) as columns:
        baseline_columns = columns["indices"]
    if not np.array_equal(baseline_columns, np.arange(len(combined_names))):
        raise ValueError(f"Fold {fold} baseline feature columns mismatch")
    baseline_review = read_json(baseline_job / "review.json")
    if baseline_review["features"] != len(combined_names):
        raise ValueError(f"Fold {fold} baseline feature count mismatch")
    return {
        "train": combined_train, "valid": combined_valid, "names": combined_names,
        "train_indices": train_indices, "valid_indices": valid_indices,
        "prepared_receipt": expected["prepared"], "selection_receipt": expected["selection"],
        "baseline_receipt": expected["baseline_model"], "baseline_job": baseline_job,
        "identity_sha256": sha(prepared / "identity.npz"),
        "mask_sha256": sha(selection / "masks.npz"),
    }


def verify_baseline_reproduction(cache: dict, tolerance: float, batch_rows: int) -> float:
    import xgboost as xgb
    baseline = xgb.XGBClassifier()
    baseline.load_model(cache["baseline_job"] / "model.ubj")
    reproduced = predict_booster(baseline.get_booster(), cache["valid"], batch_rows)
    with np.load(cache["baseline_job"] / "valid_prob.npz", allow_pickle=False) as stored:
        expected = np.asarray(stored["prob"], dtype=np.float64)
    maximum = float(np.max(np.abs(reproduced - expected)))
    if maximum >= tolerance:
        raise ValueError(f"Frozen baseline model reproduction mismatch: {maximum}")
    return maximum


def train_fold(config: dict, base_config: dict, reference: dict, cache: dict,
               fold: int, out: Path, environment: dict) -> tuple[dict[str, np.ndarray], bool]:
    import pandas as pd
    import psutil
    import xgboost as xgb

    names_hash = objsha(cache["names"])
    signature = fold_signature(config, fold, cache["identity_sha256"], names_hash, cache["mask_sha256"])
    model_root = out / "models"
    job = model_root / f"fold_{fold}"
    reused = completed_fit(job, signature)
    if reused is not None:
        emit(out, "FOLD_REUSED", fold=fold)
        return reused, True
    model_root.mkdir(exist_ok=True)
    partial = model_root / f".fold_{fold}.{os.getpid()}.partial"
    if partial.exists():
        raise ValueError(f"Current-process partial directory already exists: {partial}")
    partial.mkdir()

    runtime = base_config["runtime"]
    dense_bytes = (cache["train"].shape[0] + cache["valid"].shape[0]) * cache["train"].shape[1] * 4
    estimated = psutil.Process().memory_info().rss + dense_bytes * runtime["dense_multiplier"] + 2**30
    if estimated >= runtime["max_ram_gib"] * 2**30:
        raise ValueError("Insufficient RAM for declared dense training")
    dense_train = cache["train"].toarray(order="C")
    dense_valid = cache["valid"].toarray(order="C")
    if dense_train.dtype != np.float32 or dense_valid.dtype != np.float32:
        raise ValueError("Dense feature dtype changed")
    params = dict(config["model"])
    params.update(device=f"cuda:{runtime['gpu_index']}", n_jobs=runtime["threads"],
                  num_class=len(reference["class_order"]))
    model = xgb.XGBClassifier(**params)
    emit(out, "GPU_FIT_START", fold=fold, rows=len(cache["train_indices"]),
         features=len(cache["names"]), trees=params["n_estimators"])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(dense_train, reference["y"][cache["train_indices"]],
                  eval_set=[(dense_train, reference["y"][cache["train_indices"]]),
                            (dense_valid, reference["y"][cache["valid_indices"]])],
                  verbose=False)
    fit_seconds = time.perf_counter() - started
    booster = model.get_booster()
    effective = json.loads(booster.save_config())
    if not effective["learner"]["generic_param"]["device"].startswith("cuda"):
        raise ValueError("Training did not use CUDA")
    if booster.num_boosted_rounds() != 800 or model.classes_.tolist() != list(range(len(reference["class_order"]))):
        raise ValueError("Fitted tree budget or class order mismatch")

    probabilities = {}
    checkpoint_metrics = {}
    train_reference = make_fold_reference(reference, cache["train_indices"])
    valid_reference = make_fold_reference(reference, cache["valid_indices"])
    for prefix in config["diagnostic_prefixes"]:
        train_prob = predict_booster(booster, cache["train"], runtime["predict_batch_rows"], prefix)
        valid_prob = predict_booster(booster, cache["valid"], runtime["predict_batch_rows"], prefix)
        probabilities[f"p_{prefix}"] = valid_prob
        train_metric, _ = metric_summary(train_reference, train_prob)
        valid_metric, _ = metric_summary(valid_reference, valid_prob)
        checkpoint_metrics[str(prefix)] = {"train": train_metric, "valid": valid_metric}
    evals = model.evals_result()
    learning_rows = [{"iteration": index + 1,
                      "train_mlogloss": float(evals["validation_0"]["mlogloss"][index]),
                      "valid_mlogloss": float(evals["validation_1"]["mlogloss"][index])}
                     for index in range(800)]
    pd.DataFrame(learning_rows).to_csv(partial / "learning_curve.csv", index=False)
    save_npz(partial / "prefix_probabilities.npz", **probabilities)
    model.save_model(partial / "model_800.ubj")
    review = {
        "fold": fold, "condition": "mrmr_global", "fit_seconds": fit_seconds,
        "features": len(cache["names"]), "names_sha256": names_hash,
        "identity_sha256": cache["identity_sha256"], "mask_sha256": cache["mask_sha256"],
        "baseline_receipts": {key: cache[key] for key in
                              ("prepared_receipt", "selection_receipt", "baseline_receipt")},
        "baseline_reproduction_max_abs": verify_baseline_reproduction(
            cache, config["baseline_prediction_tolerance"], runtime["predict_batch_rows"]),
        "requested_parameters": params, "effective_parameters": effective,
        "gpu": environment["gpu"], "estimated_ram_bytes": estimated,
        "warnings": [str(item.message) for item in captured],
        "checkpoint_metrics": checkpoint_metrics,
        "selection_scope": "ADAPTIVE_DEVELOPMENT seed43; no early stopping",
        "test_read": False, "server_score_used": False, "submission_created": False,
    }
    write_json(partial / "review.json", review)
    files = {name: sha(partial / name) for name in
             ("model_800.ubj", "prefix_probabilities.npz", "learning_curve.csv", "review.json")}
    write_json(partial / "COMPLETE.json", {"signature": signature, "files": files})
    verify_complete_receipt(partial, signature)
    if job.exists():
        raise ValueError(f"Completed fold destination appeared during fit: {job}")
    partial.replace(job)
    emit(out, "GPU_FIT_COMPLETE", fold=fold, fit_seconds=fit_seconds,
         valid_macro_f1={key: value["valid"]["macro_f1"] for key, value in checkpoint_metrics.items()})
    del model, booster, dense_train, dense_valid
    gc.collect()
    return probabilities, False


def preflight(config: dict, base_config_path: Path, inputs: Path,
              repeat_root: Path, out: Path, training: bool):
    import lab_metrics as L

    validate_config(config)
    project_root = Path(__file__).resolve().parents[1]
    verify_named_hashes(project_root, config["frozen_source_hashes"], "frozen source")
    if sha(base_config_path) != config["base_config_sha256"]:
        raise ValueError("Base config hash mismatch")
    base_config = read_json(base_config_path)
    verify_named_hashes(inputs, config["input_hashes"], "training input")
    verify_named_hashes(repeat_root, config["repeat_root_hashes"], "repeat artifact")
    input_reference = L.load_oof(inputs / "mcmp_xgb.npz")
    seed_reference = L.load_oof(repeat_root / "seed_43" / "mrmr_global_oof.npz")
    for key in ("ids", "y", "groups", "class_order"):
        if not np.array_equal(input_reference[key], seed_reference[key]):
            raise ValueError(f"Seed43 reference {key} mismatch")
    with np.load(repeat_root / "seed_43" / "folds.npz", allow_pickle=False) as stored:
        if not (np.array_equal(stored["ids"], seed_reference["ids"]) and
                np.array_equal(stored["groups"], seed_reference["groups"]) and
                np.array_equal(stored["folds"], seed_reference["folds"])):
            raise ValueError("Seed43 fold receipt identity mismatch")

    source_hash = sha(Path(__file__))
    resolved_contract_signature = contract_signature(config, base_config_path)
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "CONTRACT.json"
    if contract_path.exists():
        if read_json(contract_path).get("signature") != resolved_contract_signature:
            raise ValueError("Output directory belongs to another source/configuration")
    else:
        allowed = {"execution.log", "launch.json"}
        unexpected = sorted(path.name for path in out.iterdir() if path.name not in allowed)
        if unexpected:
            raise ValueError("Output directory contains unrelated artifacts: " + ", ".join(unexpected))
        write_json(contract_path, {
            "signature": resolved_contract_signature, "config": config, "runner_sha256": source_hash,
            "base_config_sha256": sha(base_config_path),
            "scope": "seed43 adaptive development mRMR+global tree budget 200 to 800",
        })
    environment = environment_receipt(base_config, training)
    write_json(out / "environment.json", environment)
    caches = {}
    baseline_reproduction = {}
    for fold in config["folds"]:
        caches[fold] = build_fold_cache(config, base_config, repeat_root, seed_reference, fold)
        baseline_reproduction[str(fold)] = verify_baseline_reproduction(
            caches[fold], config["baseline_prediction_tolerance"],
            base_config["runtime"]["predict_batch_rows"])
    receipt = {
        "runner_sha256": source_hash, "config_sha256": sha(Path(config["config_path"])),
        "base_config_sha256": sha(base_config_path), "verified_inputs": config["input_hashes"],
        "verified_repeat_artifacts": config["repeat_root_hashes"],
        "fold_receipts": config["frozen_fold_receipts"],
        "baseline_reproduction_max_abs": baseline_reproduction,
        "rows": len(seed_reference["ids"]), "classes": len(seed_reference["class_order"]),
        "canonical_groups": len(np.unique(seed_reference["groups"])),
        "seed": 43, "train_only": True, "test_read": False,
    }
    write_json(out / "INPUT_RECEIPT.json", receipt)
    emit(out, "PREFLIGHT_COMPLETE", rows=receipt["rows"], classes=receipt["classes"],
         canonical_groups=receipt["canonical_groups"], baseline_reproduction_max_abs=baseline_reproduction)
    return base_config, seed_reference, caches, environment, resolved_contract_signature


def report(config: dict, reference: dict, predictions: dict[str, np.ndarray],
           frozen_probability: np.ndarray, out: Path, fits_executed: int,
           fits_reused: int) -> dict:
    import pandas as pd
    import lab_metrics as L

    endpoints = {"frozen_200": frozen_probability,
                 **{f"fit_prefix_{prefix}": predictions[f"p_{prefix}"]
                    for prefix in config["diagnostic_prefixes"]}}
    metrics = {}
    rows = []
    per_class = {}
    for name, probability in endpoints.items():
        score = L.metrics(reference, probability)
        metrics[name] = score
        per_class[name] = score["per_class"]
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{index}": value for index, value in enumerate(score["fold_f1"])}})
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        L.save_npz(out / f"{name}_oof.npz", **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=probability)
    pd.DataFrame(rows).to_csv(out / "endpoint_scoreboard.csv", index=False)

    comparisons = {}
    for candidate, baseline in (("fit_prefix_200", "frozen_200"),
                                ("fit_prefix_400", "frozen_200"),
                                ("fit_prefix_800", "frozen_200"),
                                ("fit_prefix_800", "fit_prefix_400"),
                                ("fit_prefix_800", "fit_prefix_200")):
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "delta_macro_f1": metrics[candidate]["macro_f1"] - metrics[baseline]["macro_f1"],
            **L.changes(reference["y"], endpoints[candidate], endpoints[baseline]),
            "bootstrap": L.bootstrap(reference, endpoints[candidate], endpoints[baseline],
                                     config["bootstrap_resamples"], 43),
        }
    baseline_class = {row["SUBCLASS"]: row for row in per_class["frozen_200"]}
    candidate_class = {row["SUBCLASS"]: row for row in per_class["fit_prefix_800"]}
    rare_rows = []
    for label in reference["class_order"]:
        label = str(label)
        if baseline_class[label]["support_rows"] <= 100:
            rare_rows.append({
                "SUBCLASS": label, "support_rows": baseline_class[label]["support_rows"],
                "frozen_200_f1": baseline_class[label]["f1"],
                "fit_prefix_800_f1": candidate_class[label]["f1"],
                "delta_f1": candidate_class[label]["f1"] - baseline_class[label]["f1"],
            })
    pd.DataFrame(rare_rows).to_csv(out / "rare_class_guard.csv", index=False)
    rare_deltas = [row["delta_f1"] for row in rare_rows]
    rare_guard = {
        "support_threshold": 100, "minimum_delta": min(rare_deltas),
        "mean_delta": float(np.mean(rare_deltas)),
        "passes_single_class_floor_minus_0_10": min(rare_deltas) >= -0.10,
        "passes_mean_floor_minus_0_02": float(np.mean(rare_deltas)) >= -0.02,
        "scope": "seed43 adaptive development screen; not independent confirmation",
    }
    compact_metrics = {name: {key: value for key, value in score.items()
                              if key not in ("per_class",)}
                       for name, score in metrics.items()}
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_TREE_BUDGET_COMPLETE",
        "seed": 43, "fit_budget": 5, "fits_executed": fits_executed,
        "fits_reused": fits_reused, "metrics": compact_metrics,
        "comparisons": comparisons, "rare_class_guard": rare_guard,
        "selection_interpretation": "200/400/800 are diagnostics on the already-used seed43 split; no early stopping",
        "seed44_read_or_fit": False, "train_only": True, "test_read": False,
        "server_score_used": False, "submission_created": False,
        "automatic_promotion": False,
    }
    write_json(out / "TREE_BUDGET_REVIEW.json", review)
    return review


def run(args) -> None:
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    config["config_path"] = str(config_path)
    out = Path(args.out).resolve()
    resolved_contract_signature = contract_signature(config, Path(args.base_config).resolve())
    if completed_run(out, resolved_contract_signature):
        print(json.dumps({"event": "COMPLETED_RUN_VERIFIED", "new_classifier_fits": 0}), flush=True)
        return
    repeat_root = Path(args.repeat_root).resolve()
    base_config, reference, caches, environment, active_contract_signature = preflight(
        config, Path(args.base_config).resolve(), Path(args.inputs).resolve(),
        repeat_root, out, training=not args.audit)
    if args.audit:
        write_json(out / "AUDIT_COMPLETE.json", {
            "status": "AUDIT_COMPLETE", "new_classifier_fits": 0,
            "contract_signature": active_contract_signature, "test_read": False,
        })
        emit(out, "AUDIT_COMPLETE", new_classifier_fits=0)
        return

    probabilities = {f"p_{prefix}": np.full((len(reference["ids"]), len(reference["class_order"])), np.nan)
                     for prefix in config["diagnostic_prefixes"]}
    coverage = np.zeros(len(reference["ids"]), dtype=np.int8)
    fits_executed = 0
    fits_reused = 0
    for fold in config["folds"]:
        fold_probability, reused = train_fold(config, base_config, reference, caches[fold],
                                              fold, out, environment)
        fits_reused += int(reused)
        fits_executed += int(not reused)
        valid_indices = caches[fold]["valid_indices"]
        for key in probabilities:
            probabilities[key][valid_indices] = fold_probability[key]
        coverage[valid_indices] += 1
        if fits_executed > config["fit_budget"]:
            raise ValueError("Classifier fit budget exceeded")
    if not np.all(coverage == 1) or any(not np.isfinite(value).all() for value in probabilities.values()):
        raise ValueError("OOF coverage is not exactly once")
    frozen = reference["prob"]
    review = report(config, reference, probabilities, frozen, out, fits_executed, fits_reused)
    artifacts = ["TREE_BUDGET_REVIEW.json", "endpoint_scoreboard.csv", "rare_class_guard.csv"]
    artifacts += [f"fit_prefix_{prefix}_oof.npz" for prefix in config["diagnostic_prefixes"]]
    artifacts += [f"fit_prefix_{prefix}_per_class.csv" for prefix in config["diagnostic_prefixes"]]
    artifacts += ["frozen_200_oof.npz", "frozen_200_per_class.csv", "INPUT_RECEIPT.json", "environment.json"]
    write_json(out / "GOAL055_COMPLETE.json", {
        "status": review["status"], "contract_signature": active_contract_signature,
        "files": {name: sha(out / name) for name in artifacts},
        "fits_executed": fits_executed, "fits_reused": fits_reused,
    })
    emit(out, "RUN_COMPLETE", fits_executed=fits_executed, fits_reused=fits_reused,
         macro_f1={name: value["macro_f1"] for name, value in review["metrics"].items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--repeat-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "FAILED.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
