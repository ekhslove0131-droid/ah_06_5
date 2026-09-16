"""Repeat the MCMP feature-view comparison on two new canonical group splits.

Each seed rebuilds fold-local vocabularies and fold-local mRMR masks, then fits
the unchanged XGBoost classifier on full, mRMR, and mRMR plus deterministic
global summaries. The fixed 50:50 mean has no learned weight.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold


CONDITIONS = ("full", "mrmr", "mrmr_global")
LAUNCHER_BOOTSTRAP_FILES = {"execution.log", "launch.json"}


def assign_group_folds(y, groups, seed, fold_count):
    y = np.asarray(y)
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or len(y) == 0:
        raise ValueError("Invalid y/groups for repeated group folds.")
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if np.any(folds[valid] != -1):
            raise ValueError("Repeated validation assignment.")
        folds[valid] = fold
    if np.any(folds < 0) or sorted(np.unique(folds).tolist()) != list(range(fold_count)):
        raise ValueError("Incomplete repeated group fold assignment.")
    assigned = {}
    for group, fold in zip(groups, folds):
        if group in assigned and assigned[group] != int(fold):
            raise ValueError("Canonical group crosses repeated folds.")
        assigned[group] = int(fold)
    classes = set(np.unique(y).tolist())
    for fold in range(fold_count):
        if set(y[folds == fold].tolist()) != classes or set(y[folds != fold].tolist()) != classes:
            raise ValueError(f"Repeated fold {fold} lacks a class.")
    return folds


def fixed_mean(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Fixed-mean probability shapes differ.")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("Fixed-mean probabilities are nonfinite.")
    if not (np.allclose(left.sum(1), 1, atol=1e-5, rtol=0) and
            np.allclose(right.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Fixed-mean inputs are not probabilities.")
    result = (left + right) / 2.0
    return result / result.sum(1, keepdims=True)


def emit(out, event, **values):
    row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), event=event, **values)
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    with (Path(out) / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def admit_output_directory(out):
    from lab_metrics import need
    out = Path(out)
    if not out.exists():
        out.mkdir(parents=True)
        return
    unexpected = sorted(path.name for path in out.iterdir()
                        if path.name not in LAUNCHER_BOOTSTRAP_FILES)
    need(not unexpected, "Output directory contains experiment artifacts: " + ", ".join(unexpected))


def write_gzip(path, value):
    partial = Path(str(path) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    partial.replace(path)


def check_sources(config, base_config_path, inputs, source_run, summary_run):
    import cbj_lab as B
    import lab_metrics as L
    if config["version"] != "cbj-windows-repeat-v1":
        raise ValueError("Wrong repeat experiment version.")
    for name, digest in config["source_hashes"].items():
        L.need(L.sha(Path(__file__).parent / name) == digest, "Source hash mismatch: " + name)
    L.need(L.sha(base_config_path) == config["base_config_sha256"], "Base configuration changed.")
    base_config = L.read_json(base_config_path)
    reference = B.check_inputs(base_config, inputs)
    markers = [
        (Path(source_run) / "EXPERIMENT_COMPLETE.json", config["source_experiment_complete_sha256"]),
        (Path(summary_run) / "SUMMARY_COMPLETE.json", config["source_summary_complete_sha256"]),
    ]
    for path, digest in markers:
        L.need(path.is_file() and L.sha(path) == digest, "Source run marker mismatch: " + path.name)
    original_full = L.load_oof(Path(source_run) / "full_oof.npz")
    original_mrmr = L.load_oof(Path(source_run) / "mrmr_oof.npz")
    original_global = L.load_oof(Path(summary_run) / "mrmr_global_oof.npz")
    for candidate in (original_full, original_mrmr, original_global):
        L.align(reference, candidate)
    return base_config, reference, original_full, original_mrmr, original_global


def prepare_fold(config, base_config, rows, reference, folds, seed, fold, out):
    import mcmp_repr as M
    import cbj_lab as B
    import lab_metrics as L
    signature = L.objsha(dict(seed=seed, fold=fold, folds_sha256=L.objsha(folds.tolist()),
        representation=base_config["representation"], source_genes=base_config["source_genes"],
        parser=config["source_hashes"]["mcmp_repr.py"]))
    out = Path(out)
    if B.valid(out, signature):
        return
    out.mkdir(parents=True, exist_ok=True)
    train_indices = np.flatnonzero(folds != fold)
    valid_indices = np.flatnonzero(folds == fold)
    L.need(not set(reference["groups"][train_indices]) & set(reference["groups"][valid_indices]),
           "Canonical group leakage in repeated fold.")
    encoder = M.Encoder(base_config["representation"]["blocks"],
                        base_config["representation"]["min_group_support"])
    train_matrix = encoder.fit_transform([rows[i] for i in train_indices], reference["groups"][train_indices])
    valid_matrix = encoder.transform([rows[i] for i in valid_indices])
    sparse.save_npz(out / "train.npz", train_matrix)
    sparse.save_npz(out / "valid.npz", valid_matrix)
    write_gzip(out / "encoder.json.gz", encoder.to_dict())
    L.save_npz(out / "identity.npz", train_indices=train_indices, valid_indices=valid_indices)
    L.write_json(out / "review.json", dict(seed=seed, fold=fold, train_rows=len(train_indices),
        valid_rows=len(valid_indices), features=len(encoder.names), names_sha256=L.objsha(encoder.names),
        train_ids_sha256=L.objsha(reference["ids"][train_indices].tolist()),
        valid_ids_sha256=L.objsha(reference["ids"][valid_indices].tolist()),
        fit_scope="fold training patients only", test_read=False))
    B.manifest(out, signature, ["train.npz", "valid.npz", "encoder.json.gz", "identity.npz", "review.json"])
    emit(out.parent.parent.parent, "FOLD_PREPARED", seed=seed, fold=fold, features=len(encoder.names))


def report_seed(config, reference, folds, predictions, out, seed):
    import lab_metrics as L
    seed_reference = dict(reference)
    seed_reference["folds"] = folds
    predictions["fixed_mean"] = fixed_mean(predictions["mrmr"], predictions["mrmr_global"])
    summaries = {name:L.metrics(seed_reference, probability) for name,probability in predictions.items()}
    rows = []
    for name, score in summaries.items():
        rows.append(dict(seed=seed, condition=name,
            **{key:value for key,value in score.items() if key not in ("per_class", "fold_f1")},
            **{f"fold_{i}":value for i,value in enumerate(score["fold_f1"])}))
        L.save_npz(Path(out) / f"{name}_oof.npz", **{key:seed_reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=predictions[name])
        pd.DataFrame(score["per_class"]).to_csv(Path(out) / f"{name}_per_class.csv", index=False)
    pd.DataFrame(rows).to_csv(Path(out) / "scoreboard.csv", index=False)
    comparisons = {}
    for name, base in (("mrmr", "full"), ("mrmr_global", "mrmr"),
                       ("fixed_mean", "mrmr"), ("fixed_mean", "mrmr_global")):
        comparisons[name + "_vs_" + base] = dict(
            delta_macro_f1=summaries[name]["macro_f1"] - summaries[base]["macro_f1"],
            **L.changes(reference["y"], predictions[name], predictions[base]),
            bootstrap=L.bootstrap(seed_reference, predictions[name], predictions[base],
                                  config["bootstrap_resamples"], seed))
    review = dict(seed=seed, summaries=summaries, comparisons=comparisons,
                  canonical_group_fold=True, train_only=True, test_read=False,
                  server_score_used=False, submission_created=False)
    L.write_json(Path(out) / "review.json", review)
    return review, rows


def run(config_path, base_config_path, inputs, source_run, summary_run, out):
    import cbj_lab as B
    import cbj_summary_view as S
    import lab_metrics as L
    import mcmp_repr as M
    config = L.read_json(config_path)
    out = Path(out)
    admit_output_directory(out)
    base_config, reference, old_full, old_mrmr, old_global = check_sources(
        config, Path(base_config_path), Path(inputs), Path(source_run), Path(summary_run))
    train_config = dict(base_config)
    train_config["source_hashes"] = dict(base_config["source_hashes"], **config["source_hashes"])
    environment = B.environment(train_config, True)
    L.write_json(out / "environment.json", environment)
    L.write_json(out / "CONTRACT.json", dict(signature=L.objsha(config), config=config,
        scope="two new canonical group splits; full, mRMR, mRMR global; fixed mean"))
    train = pd.read_csv(Path(inputs) / base_config["inputs"]["train"]["filename"], dtype=str)
    L.need(train.columns.tolist() == ["ID", "SUBCLASS"] + base_config["source_genes"],
           "Raw schema or gene order mismatch.")
    L.need(train.ID.tolist() == reference["ids"].tolist(), "Raw train ID order mismatch.")
    L.need(train.SUBCLASS.tolist() == reference["class_order"][reference["y"]].tolist(),
           "Raw train labels mismatch.")
    emit(out, "PARSER_START", rows=len(train), genes=len(base_config["source_genes"]))
    feature_rows, parser_stats = M.build_feature_rows(train, base_config["source_genes"])
    del train
    gc.collect()
    all_rows = []
    all_reviews = {}
    for seed in config["seeds"]:
        seed_out = out / f"seed_{seed}"
        seed_out.mkdir()
        folds = assign_group_folds(reference["y"], reference["groups"], seed, config["fold_count"])
        L.save_npz(seed_out / "folds.npz", ids=reference["ids"], groups=reference["groups"], folds=folds)
        L.write_json(seed_out / "fold_review.json", dict(seed=seed,
            fold_sizes={str(fold):int((folds == fold).sum()) for fold in range(config["fold_count"])},
            folds_sha256=L.objsha(folds.tolist()), parser_stats=parser_stats,
            canonical_groups=len(np.unique(reference["groups"])), test_read=False))
        for fold in range(config["fold_count"]):
            prepare_fold(config, base_config, feature_rows, reference, folds, seed, fold,
                         seed_out / "prepared" / f"fold_{fold}")
        predictions = {name:np.full_like(reference["prob"], np.nan) for name in CONDITIONS}
        seen = np.zeros(len(reference["ids"]), dtype=np.int8)
        for fold in range(config["fold_count"]):
            prepared = seed_out / "prepared" / f"fold_{fold}"
            train_matrix = sparse.load_npz(prepared / "train.npz").tocsr()
            valid_matrix = sparse.load_npz(prepared / "valid.npz").tocsr()
            names = B.read_gzip(prepared / "encoder.json.gz")["names"]
            with np.load(prepared / "identity.npz", allow_pickle=False) as identity:
                train_indices, valid_indices = identity["train_indices"], identity["valid_indices"]
            masks = B.selection(train_config, train_matrix, names, reference["y"][train_indices],
                reference["ids"][train_indices].tolist(), seed_out / "selection" / f"fold_{fold}",
                fold, seed_out)
            global_train, global_names, _, _ = S.build_summary_views(
                train_matrix, names, {"placeholder":{"__not_a_gene__"}}, len(base_config["source_genes"]))
            global_valid, valid_global_names, _, _ = S.build_summary_views(
                valid_matrix, names, {"placeholder":{"__not_a_gene__"}}, len(base_config["source_genes"]))
            L.need(global_names == valid_global_names, "Global summary order mismatch.")
            mrmr_train, mrmr_valid = train_matrix[:, masks["mrmr"]], valid_matrix[:, masks["mrmr"]]
            combined_train = sparse.hstack([mrmr_train, global_train], format="csr", dtype=np.float32)
            combined_valid = sparse.hstack([mrmr_valid, global_valid], format="csr", dtype=np.float32)
            combined_names = [names[i] for i in masks["mrmr"]] + global_names
            predictions["full"][valid_indices] = B.train_one(
                train_config, train_matrix, valid_matrix, np.arange(len(names)), reference["y"][train_indices],
                seed_out / "models" / "full" / f"fold_{fold}", fold, f"seed{seed}_full",
                environment, seed_out, names)
            predictions["mrmr"][valid_indices] = B.train_one(
                train_config, train_matrix, valid_matrix, masks["mrmr"], reference["y"][train_indices],
                seed_out / "models" / "mrmr" / f"fold_{fold}", fold, f"seed{seed}_mrmr",
                environment, seed_out, names)
            predictions["mrmr_global"][valid_indices] = B.train_one(
                train_config, combined_train, combined_valid, np.arange(len(combined_names)),
                reference["y"][train_indices], seed_out / "models" / "mrmr_global" / f"fold_{fold}",
                fold, f"seed{seed}_mrmr_global", environment, seed_out, combined_names)
            seen[valid_indices] += 1
            del train_matrix, valid_matrix, mrmr_train, mrmr_valid, combined_train, combined_valid
            gc.collect()
        L.need(np.all(seen == 1), "Repeated OOF coverage must be exactly once.")
        review, rows = report_seed(config, reference, folds, predictions, seed_out, seed)
        all_reviews[str(seed)] = review
        all_rows.extend(rows)
        emit(out, "SEED_COMPLETE", seed=seed,
             scores={name:value["macro_f1"] for name,value in review["summaries"].items()})
    del feature_rows
    gc.collect()
    original_reference = dict(reference)
    original_predictions = {
        "full":old_full["prob"], "mrmr":old_mrmr["prob"], "mrmr_global":old_global["prob"],
        "fixed_mean":fixed_mean(old_mrmr["prob"], old_global["prob"]),
    }
    original_scores = {name:L.metrics(original_reference, probability) for name,probability in original_predictions.items()}
    combined = [dict(split="original_fixed", condition=name, macro_f1=value["macro_f1"])
                for name,value in original_scores.items()]
    for seed, review in all_reviews.items():
        combined.extend(dict(split=f"seed_{seed}", condition=name, macro_f1=value["macro_f1"])
                        for name,value in review["summaries"].items())
    combined_frame = pd.DataFrame(combined)
    combined_frame.to_csv(out / "three_split_scores.csv", index=False)
    pivot = combined_frame.pivot(index="split", columns="condition", values="macro_f1")
    deltas = pd.DataFrame({
        "mrmr_vs_full":pivot["mrmr"] - pivot["full"],
        "global_vs_mrmr":pivot["mrmr_global"] - pivot["mrmr"],
        "fixed_mean_vs_mrmr":pivot["fixed_mean"] - pivot["mrmr"],
        "fixed_mean_vs_global":pivot["fixed_mean"] - pivot["mrmr_global"],
    })
    deltas.to_csv(out / "three_split_deltas.csv")
    gate = dict(
        required_wins=config["promotion_gate"]["required_wins_of_three"],
        observed_wins={column:int((deltas[column] > 0).sum()) for column in deltas},
        mean_delta={column:float(deltas[column].mean()) for column in deltas},
        median_delta={column:float(deltas[column].median()) for column in deltas},
        automatic_promotion=False,
        interpretation="Development robustness screen; final stacking still requires whole-stack nested validation.",
    )
    final = dict(status="DEVELOPMENT_REPEAT_CV_COMPLETE", seeds=config["seeds"],
        new_gpu_fits=config["fit_budget"], seed_reviews=all_reviews, original_scores=original_scores,
        gate=gate, train_only=True, test_read=False, server_score_used=False,
        submission_created=False, automatic_promotion=False)
    L.write_json(out / "repeat_review.json", final)
    check_sources(config, Path(base_config_path), Path(inputs), Path(source_run), Path(summary_run))
    L.write_json(out / "REPEAT_COMPLETE.json", dict(status=final["status"],
        review_sha256=L.sha(out / "repeat_review.json"),
        scores_sha256=L.sha(out / "three_split_scores.csv"), deltas_sha256=L.sha(out / "three_split_deltas.csv")))
    emit(out, "RUN_COMPLETE", gate=gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--summary-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        run(args.config, args.base_config, args.inputs, args.source_run, args.summary_run, args.out)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        from lab_metrics import write_json
        write_json(out / "FAILED.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
