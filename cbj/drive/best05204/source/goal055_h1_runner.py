"""Seed43 train-only H1 runner: append 12 label-free raw annotation counts."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import goal055_tree_budget as T
import goal055_class_weight as W
import goal055_raw_annotation as R


def h1_contract_signature(config: dict, baseline_oof: Path) -> str:
    return T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                     "baseline_oof_sha256": T.sha(baseline_oof)})


def completed_h1_run(out: Path, signature: str) -> bool:
    marker = Path(out) / "H1_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker)
    if receipt.get("contract_signature") != signature:
        raise ValueError("Completed H1 run belongs to another configuration")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Malformed H1 completion receipt")
    for name, digest in files.items():
        path = Path(out) / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Completed H1 integrity failure: {path}")
    return True


def preflight(config: dict, base_config_path: Path, inputs: Path,
              repeat_root: Path, baseline_oof: Path, out: Path, training: bool):
    import lab_metrics as L

    R.validate_h1_config(config)
    helper_paths = {
        "tree_helper_sha256": Path(T.__file__),
        "weight_helper_sha256": Path(W.__file__),
        "raw_builder_sha256": Path(R.__file__),
    }
    for key, path in helper_paths.items():
        if T.sha(path) != config.get(key):
            raise ValueError(f"Pinned H1 helper mismatch: {key}")
    T.verify_named_hashes(T.PROJECT_ROOT, config["frozen_source_hashes"], "frozen source")
    if T.sha(base_config_path) != config["base_config_sha256"]:
        raise ValueError("H1 base config hash mismatch")
    T.verify_named_hashes(inputs, config["input_hashes"], "training input")
    T.verify_named_hashes(repeat_root, config["repeat_root_hashes"], "repeat artifact")
    if T.sha(baseline_oof) != config["baseline_oof_sha256"]:
        raise ValueError("H1 alpha0.5 baseline OOF hash mismatch")
    base_config = T.read_json(base_config_path)
    reference = L.load_oof(repeat_root / "seed_43" / "mrmr_global_oof.npz")
    baseline = L.load_oof(baseline_oof)
    for key in ("ids", "y", "groups", "folds", "class_order"):
        if not np.array_equal(reference[key], baseline[key]):
            raise ValueError(f"H1 baseline {key} mismatch")

    train = pd.read_csv(inputs / base_config["inputs"]["train"]["filename"], dtype=str)
    expected_columns = ["ID", "SUBCLASS"] + base_config["source_genes"]
    if train.columns.tolist() != expected_columns:
        raise ValueError("H1 raw training schema/gene order mismatch")
    if train.ID.tolist() != reference["ids"].tolist():
        raise ValueError("H1 raw ID order mismatch")
    if train.SUBCLASS.tolist() != reference["class_order"][reference["y"]].tolist():
        raise ValueError("H1 raw target identity mismatch")
    raw_view, raw_names, raw_stats = R.build_raw_annotation_view(
        train[base_config["source_genes"]], base_config["source_genes"])
    del train
    gc.collect()
    if raw_names != config["raw_feature_names"] or raw_view.shape != (len(reference["ids"]), 12):
        raise ValueError("H1 raw view shape/order mismatch")
    R.validate_raw_stats(raw_stats, config["expected_raw_stats"])

    signature = h1_contract_signature(config, baseline_oof)
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "CONTRACT.json"
    if contract_path.exists():
        if T.read_json(contract_path).get("signature") != signature:
            raise ValueError("H1 output directory belongs to another configuration")
    else:
        unexpected = sorted(path.name for path in out.iterdir()
                            if path.name not in {"launch.json", "execution.log"})
        if unexpected:
            raise ValueError("H1 output directory contains unrelated artifacts")
        T.write_json(contract_path, {
            "signature": signature, "config": config, "runner_sha256": T.sha(Path(__file__)),
            "baseline_oof": str(baseline_oof),
            "scope": "seed43 alpha0.5 mRMR-global plus exactly 12 raw annotation counts",
        })
    environment = T.environment_receipt(base_config, training)
    T.write_json(out / "environment.json", environment)
    T.save_npz(out / "raw_annotation_view.npz", values=raw_view.toarray())
    T.write_json(out / "RAW_VIEW_RECEIPT.json", {
        **raw_stats, "names": raw_names, "names_sha256": T.objsha(raw_names),
        "matrix_sha256": T.sha(out / "raw_annotation_view.npz"),
        "input_train_sha256": config["input_hashes"]["train.csv"],
        "id_order_sha256": T.objsha(reference["ids"].tolist()),
        "target_used_for_features": False,
    })
    caches = {}
    for fold in config["folds"]:
        base_cache = T.build_fold_cache(config, base_config, repeat_root, reference, fold)
        caches[fold] = R.append_raw_view(base_cache, raw_view, raw_names)
    T.emit(out, "H1_PREFLIGHT_COMPLETE", rows=len(reference["ids"]), raw_features=12,
           total_tokens=raw_stats["total_tokens"], test_read=False)
    return base_config, reference, baseline, caches, environment, signature


def report(config: dict, reference: dict, baseline: dict, probability,
           out: Path, executed: int, reused: int):
    import lab_metrics as L

    endpoints = {"alpha_0p5_baseline": baseline["prob"], "h1_raw12": probability}
    scores = {name: L.metrics(reference, value) for name, value in endpoints.items()}
    rows = []
    for name, score in scores.items():
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])}})
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        L.save_npz(out / f"{name}_oof.npz", **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=endpoints[name])
    pd.DataFrame(rows).to_csv(out / "h1_scoreboard.csv", index=False)
    base_classes = {row["SUBCLASS"]: row for row in scores["alpha_0p5_baseline"]["per_class"]}
    h1_classes = {row["SUBCLASS"]: row for row in scores["h1_raw12"]["per_class"]}
    rare = [{"SUBCLASS": label, "support_rows": base_classes[label]["support_rows"],
             "baseline_f1": base_classes[label]["f1"], "h1_f1": h1_classes[label]["f1"],
             "delta_f1": h1_classes[label]["f1"] - base_classes[label]["f1"]}
            for label in map(str, reference["class_order"])
            if base_classes[label]["support_rows"] <= 100]
    pd.DataFrame(rare).to_csv(out / "rare_class_guard.csv", index=False)
    deltas = [row["delta_f1"] for row in rare]
    comparison = {
        "delta_macro_f1": scores["h1_raw12"]["macro_f1"] - scores["alpha_0p5_baseline"]["macro_f1"],
        **L.changes(reference["y"], probability, baseline["prob"]),
        "bootstrap": L.bootstrap(reference, probability, baseline["prob"],
                                 config["bootstrap_resamples"], 43),
        "rare_min_delta": min(deltas), "rare_mean_delta": float(np.mean(deltas)),
    }
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_H1_RAW12_COMPLETE", "seed": 43,
        "scores": {name: {key: value for key, value in score.items() if key != "per_class"}
                   for name, score in scores.items()},
        "comparison": comparison, "fits_executed": executed, "fits_reused": reused,
        "fit_budget": 5, "weight_alpha": 0.5, "raw_feature_count": 12,
        "feature_scope": "deterministic raw train annotations; no labels",
        "automatic_promotion": False, "test_read": False,
        "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "H1_REVIEW.json", review)
    return review


def run(args):
    config = T.read_json(Path(args.config))
    out = Path(args.out).resolve()
    signature = h1_contract_signature(config, Path(args.baseline_oof).resolve())
    if completed_h1_run(out, signature):
        print(json.dumps({"event": "COMPLETED_H1_RUN_VERIFIED", "new_classifier_fits": 0}), flush=True)
        return
    base_config, reference, baseline, caches, environment, signature = preflight(
        config, Path(args.base_config).resolve(), Path(args.inputs).resolve(),
        Path(args.repeat_root).resolve(), Path(args.baseline_oof).resolve(), out,
        training=not args.audit)
    if args.audit:
        T.write_json(out / "AUDIT_COMPLETE.json", {"status": "AUDIT_COMPLETE", "new_fits": 0})
        return
    prediction = np.full_like(reference["prob"], np.nan)
    coverage = np.zeros(len(reference["ids"]), dtype=np.int8)
    executed = reused = 0
    for fold in config["folds"]:
        fold_probability, was_reused = W.train_weight_fold(
            config, base_config, reference, caches[fold], fold,
            config["weight_alpha"], out, environment)
        indices = caches[fold]["valid_indices"]
        prediction[indices] = fold_probability
        coverage[indices] += 1
        executed += int(not was_reused)
        reused += int(was_reused)
    if executed > 5 or not np.all(coverage == 1) or not np.isfinite(prediction).all():
        raise ValueError("H1 fit budget or OOF coverage failure")
    review = report(config, reference, baseline, prediction, out, executed, reused)
    files = ["H1_REVIEW.json", "h1_scoreboard.csv", "rare_class_guard.csv",
             "alpha_0p5_baseline_oof.npz", "alpha_0p5_baseline_per_class.csv",
             "h1_raw12_oof.npz", "h1_raw12_per_class.csv", "environment.json",
             "raw_annotation_view.npz", "RAW_VIEW_RECEIPT.json"]
    T.write_json(out / "H1_COMPLETE.json", {
        "status": review["status"], "contract_signature": signature,
        "files": {name: T.sha(out / name) for name in files}, "new_gpu_fits": executed,
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
