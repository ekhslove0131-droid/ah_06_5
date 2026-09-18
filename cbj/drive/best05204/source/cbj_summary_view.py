"""Deterministic global and pathway summaries for the fixed MCMP representation.

The summaries are calculated from each fold's already encoded train/validation
matrix. They preserve the existing gene and mutation-detail features and add no
test-derived vocabulary, labels, or inferred biological activity.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse

import cbj_lab as B
import lab_metrics as L


TYPE_KINDS = (
    "missense", "synonymous", "frameshift", "stop_gained", "stop_lost",
    "deletion", "insertion", "delins", "complex_unparsed",
)
CONDITIONS = ("mrmr_global", "mrmr_global_path")
LAUNCHER_BOOTSTRAP_FILES = {"execution.log", "launch.json"}


def emit(out, event, **values):
    row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), event=event, **values)
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    with (Path(out) / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def admit_output_directory(out):
    out = Path(out)
    if not out.exists():
        out.mkdir(parents=True)
        return
    unexpected = sorted(path.name for path in out.iterdir()
                        if path.name not in LAUNCHER_BOOTSTRAP_FILES)
    L.need(not unexpected, "Output directory contains experiment artifacts: " + ", ".join(unexpected))


def _sum_columns(matrix, indices):
    if not indices:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.asarray(matrix[:, indices].sum(axis=1)).ravel().astype(np.float32)


def _column(matrix, index, name):
    if name not in index:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.asarray(matrix[:, index[name]].toarray()).ravel().astype(np.float32)


def _safe_ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32),
                     where=denominator > 0).astype(np.float32)


def build_summary_views(matrix, names, stage_genes, source_gene_count):
    """Return deterministic global and pathway CSR views with fixed column order."""
    L.need(sparse.isspmatrix_csr(matrix), "Summary input must be CSR.")
    L.need(matrix.shape[1] == len(names), "Summary feature names are misaligned.")
    index = {name:i for i,name in enumerate(names)}
    mutation = _column(matrix, index, "global|mutation_gene_count")
    missing = _column(matrix, index, "global|missing_gene_count")
    observed = _column(matrix, index, "global|observed_gene_count")
    type_counts = {}
    for kind in TYPE_KINDS:
        type_counts[kind] = _sum_columns(matrix, [i for i,name in enumerate(names)
                                                   if name.startswith("type|") and name.endswith("|" + kind)])
    total = sum(type_counts.values(), np.zeros(matrix.shape[0], dtype=np.float32))
    truncating = type_counts["frameshift"] + type_counts["stop_gained"]
    indel = type_counts["deletion"] + type_counts["insertion"] + type_counts["delins"]
    parsed = total - type_counts["complex_unparsed"]
    global_columns = [
        ("summary|log1p_mutated_gene_count", np.log1p(mutation).astype(np.float32)),
        ("summary|mutation_fraction_observed", _safe_ratio(mutation, observed)),
        ("summary|missing_fraction_panel", missing / float(source_gene_count)),
        ("summary|type|total_token_count", total),
        ("summary|type|multi_annotation_excess", np.maximum(total - mutation, 0)),
        ("summary|type|multi_annotation_fraction", _safe_ratio(np.maximum(total - mutation, 0), total)),
        ("summary|type|parsed_fraction", _safe_ratio(parsed, total)),
        ("summary|type|truncating_count", truncating),
        ("summary|type|truncating_fraction", _safe_ratio(truncating, total)),
        ("summary|type|indel_count", indel),
        ("summary|type|indel_fraction", _safe_ratio(indel, total)),
    ]
    for kind in TYPE_KINDS:
        global_columns.append((f"summary|type|{kind}_count", type_counts[kind]))
        global_columns.append((f"summary|type|{kind}_fraction", _safe_ratio(type_counts[kind], total)))

    path_columns = []
    for stage in sorted(stage_genes):
        genes = sorted(set(stage_genes[stage]))
        L.need(genes, "Empty pathway stage: " + stage)
        mutated = _sum_columns(matrix, [index[name] for gene in genes
                               if (name := f"gene|{gene}|nonWT") in index])
        stage_missing = _sum_columns(matrix, [index[name] for gene in genes
                                    if (name := f"gene|{gene}|missing") in index])
        path_columns.extend([
            (f"path_summary|{stage}|mutated_gene_count", mutated),
            (f"path_summary|{stage}|any_mutated", (mutated > 0).astype(np.float32)),
            (f"path_summary|{stage}|mutated_gene_fraction", mutated / float(len(genes))),
            (f"path_summary|{stage}|mutation_share_of_patient", _safe_ratio(mutated, mutation)),
            (f"path_summary|{stage}|missing_gene_count", stage_missing),
            (f"path_summary|{stage}|missing_gene_fraction", stage_missing / float(len(genes))),
        ])
        for kind in TYPE_KINDS:
            indices = [index[name] for gene in genes
                       if (name := f"type|{gene}|{kind}") in index]
            path_columns.append((f"path_summary|{stage}|type|{kind}_count",
                                 _sum_columns(matrix, indices)))

    def pack(columns):
        values = np.column_stack([value for _,value in columns]).astype(np.float32, copy=False)
        L.need(np.isfinite(values).all() and (values >= 0).all(), "Summary produced invalid values.")
        return sparse.csr_matrix(values), [name for name,_ in columns]

    global_view, global_names = pack(global_columns)
    path_view, path_names = pack(path_columns)
    return global_view, global_names, path_view, path_names


def load_stage_genes(path, expected_stages):
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    L.need({"gene", "stage", "present_in_train"} <= set(frame.columns), "Invalid pathway mapping schema.")
    frame = frame[frame["present_in_train"].str.lower().eq("true")]
    mapping = {stage:set(frame.loc[frame.stage.eq(stage), "gene"].dropna()) for stage in expected_stages}
    L.need(all(mapping.values()), "A configured pathway stage has no mapped train genes.")
    return mapping


def source_mrmr_columns(source, fold, feature_count):
    job = Path(source) / "selection" / f"fold_{fold}"
    receipt = L.read_json(job / "COMPLETE.json")
    L.need(receipt["files"]["masks.npz"] == L.sha(job / "masks.npz"), "mRMR mask integrity failure.")
    with np.load(job / "masks.npz", allow_pickle=False) as data:
        take = np.asarray(data["mrmr"], dtype=np.int64)
    L.need(take.ndim == 1 and len(np.unique(take)) == len(take), "Invalid mRMR mask.")
    L.need((take >= 0).all() and (take < feature_count).all(), "mRMR mask index out of range.")
    return take


def check_source(config, base_config_path, inputs, source):
    L.need(config["version"] == "cbj-windows-summary-v1", "Wrong summary experiment version.")
    for name, digest in config["source_hashes"].items():
        L.need(L.sha(Path(__file__).parent / name) == digest, "Source hash mismatch: " + name)
    L.need(L.sha(base_config_path) == config["base_config_sha256"], "Base configuration changed.")
    base_config = L.read_json(base_config_path)
    reference = B.check_inputs(base_config, inputs)
    marker = Path(source) / "EXPERIMENT_COMPLETE.json"
    L.need(marker.is_file() and L.sha(marker) == config["source_experiment_complete_sha256"],
           "Completed source experiment marker mismatch.")
    return base_config, reference


def report(config, reference, source, predictions, out):
    baseline = L.load_oof(Path(source) / "mrmr_oof.npz")
    L.align(reference, baseline)
    summaries, comparisons, rows = {}, {}, []
    for condition, prediction in predictions.items():
        score = L.metrics(reference, prediction)
        summaries[condition] = score
        rows.append(dict(condition=condition, **{key:value for key,value in score.items()
                    if key not in ("per_class", "fold_f1")},
                    **{f"fold_{i}":value for i,value in zip(config["folds"], score["fold_f1"])},
                    delta_vs_mrmr=score["macro_f1"] - L.metrics(reference, baseline["prob"])["macro_f1"]))
        pd.DataFrame(score["per_class"]).to_csv(Path(out) / f"{condition}_per_class.csv", index=False)
        L.save_npz(Path(out) / f"{condition}_oof.npz", **{key:reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=prediction)
        comparisons[condition + "_vs_mrmr"] = dict(
            delta_macro_f1=score["macro_f1"] - L.metrics(reference, baseline["prob"])["macro_f1"],
            **L.changes(reference["y"], prediction, baseline["prob"]),
            bootstrap=L.bootstrap(reference, prediction, baseline["prob"], config["bootstrap_resamples"], 42))
    pd.DataFrame(rows).to_csv(Path(out) / "summary_scoreboard.csv", index=False)
    review = dict(status="DEVELOPMENT_SUMMARY_CV_COMPLETE", summaries=summaries, comparisons=comparisons,
                  same_canonical_folds=True, train_only=True, test_read=False, server_score_used=False,
                  submission_created=False, automatic_promotion=False, whole_stack_nested_verified=False)
    L.write_json(Path(out) / "summary_review.json", review)
    emit(out, "RUN_COMPLETE", scores={key:value["macro_f1"] for key,value in summaries.items()})
    return review


def run(config_path, base_config_path, inputs, source, out):
    config = L.read_json(config_path)
    out, source = Path(out), Path(source)
    admit_output_directory(out)
    base_config, reference = check_source(config, Path(base_config_path), Path(inputs), source)
    stage_genes = load_stage_genes(Path(__file__).parent / config["mapping_file"], config["stages"])
    train_config = dict(base_config)
    train_config["source_hashes"] = dict(base_config["source_hashes"], **config["source_hashes"])
    environment = B.environment(train_config, True)
    L.write_json(out / "environment.json", environment)
    L.write_json(out / "CONTRACT.json", dict(signature=L.objsha(config), config=config,
                 source_run=str(source), scope="same-fold deterministic global and pathway views"))
    predictions = {name:np.full_like(reference["prob"], np.nan) for name in CONDITIONS}
    seen = np.zeros(len(reference["ids"]), dtype=np.int8)
    for fold in config["folds"]:
        prepared = source / "prepared" / f"fold_{fold}"
        matrix_train = sparse.load_npz(prepared / "train.npz").tocsr()
        matrix_valid = sparse.load_npz(prepared / "valid.npz").tocsr()
        names = B.read_gzip(prepared / "encoder.json.gz")["names"]
        with np.load(prepared / "identity.npz", allow_pickle=False) as identity:
            train_indices, valid_indices = identity["train_indices"], identity["valid_indices"]
        L.need(np.array_equal(valid_indices, np.flatnonzero(reference["folds"] == fold)),
               "Prepared validation identity mismatch.")
        take = source_mrmr_columns(source, fold, len(names))
        global_train, global_names, path_train, path_names = build_summary_views(
            matrix_train, names, stage_genes, config["source_gene_count"])
        global_valid, valid_global_names, path_valid, valid_path_names = build_summary_views(
            matrix_valid, names, stage_genes, config["source_gene_count"])
        L.need(global_names == valid_global_names and path_names == valid_path_names,
               "Summary train/validation feature order mismatch.")
        base_train, base_valid = matrix_train[:, take], matrix_valid[:, take]
        matrices = {
            "mrmr_global": (
                sparse.hstack([base_train, global_train], format="csr", dtype=np.float32),
                sparse.hstack([base_valid, global_valid], format="csr", dtype=np.float32),
                [names[i] for i in take] + global_names,
            ),
            "mrmr_global_path": (
                sparse.hstack([base_train, global_train, path_train], format="csr", dtype=np.float32),
                sparse.hstack([base_valid, global_valid, path_valid], format="csr", dtype=np.float32),
                [names[i] for i in take] + global_names + path_names,
            ),
        }
        for condition, (fold_train, fold_valid, feature_names) in matrices.items():
            columns = np.arange(len(feature_names), dtype=np.int64)
            predictions[condition][valid_indices] = B.train_one(
                train_config, fold_train, fold_valid, columns, reference["y"][train_indices],
                out / "models" / condition / f"fold_{fold}", fold, condition,
                environment, out, feature_names)
        seen[valid_indices] += 1
        del matrix_train, matrix_valid, matrices, base_train, base_valid
        gc.collect()
    L.need(np.all(seen == 1), "OOF coverage must be exactly once.")
    check_source(config, Path(base_config_path), Path(inputs), source)
    review = report(config, reference, source, predictions, out)
    L.write_json(out / "SUMMARY_COMPLETE.json", dict(status=review["status"],
        review_sha256=L.sha(out / "summary_review.json"),
        oof_sha256={name:L.sha(out / f"{name}_oof.npz") for name in CONDITIONS}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        run(args.config, args.base_config, args.inputs, args.source, args.out)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        L.write_json(out / "FAILED.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
