"""No-fit support-guarded fixed mean of frozen H1 and C10 OOF probabilities."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import goal055_tree_budget as T
import goal055_tfidf_logistic as F


def _numeric_zero_rows(matrix):
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return np.diff(matrix.indptr) == 0


def support_guard_mask(train_symbolic, valid_symbolic):
    if not sparse.issparse(train_symbolic) or not sparse.issparse(valid_symbolic):
        raise ValueError("Support guard requires sparse symbolic matrices")
    if train_symbolic.shape[1] != valid_symbolic.shape[1]:
        raise ValueError("Support guard train/valid columns differ")
    train_zero = _numeric_zero_rows(train_symbolic)
    valid_zero = _numeric_zero_rows(valid_symbolic)
    mask = valid_zero.copy() if not train_zero.any() else np.zeros(len(valid_zero), dtype=bool)
    receipt = {
        "train_rows": train_symbolic.shape[0], "valid_rows": valid_symbolic.shape[0],
        "symbolic_features": train_symbolic.shape[1],
        "train_zero_rows": int(train_zero.sum()), "valid_zero_rows": int(valid_zero.sum()),
        "guarded_rows": int(mask.sum()),
        "rule": "guard valid numeric-allzero iff train numeric-allzero count equals zero",
        "labels_or_ids_accepted_by_rule": False,
    }
    return mask, receipt


def _probability(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 2 or len(value) == 0 or value.shape[1] < 2 or
            not np.isfinite(value).all() or (value < 0).any() or
            not np.allclose(value.sum(1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("Invalid source probability matrix")
    return value / value.sum(1, keepdims=True)


def guarded_fixed_mean(h1_probability, c10_probability, guard_mask):
    h1 = _probability(h1_probability)
    c10 = _probability(c10_probability)
    guard_mask = np.asarray(guard_mask, dtype=bool)
    if h1.shape != c10.shape or guard_mask.shape != (len(h1),):
        raise ValueError("Guarded mean shape mismatch")
    result = (h1 + c10) / 2.0
    result[guard_mask] = h1[guard_mask]
    result /= result.sum(1, keepdims=True)
    return result


def validate_exact_alignment(reference: dict, candidate: dict) -> None:
    for key in ("ids", "y", "groups", "folds", "class_order"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Exact OOF {key} alignment mismatch")


def completed_run(out: Path, signature: str) -> bool:
    marker = Path(out) / "SUPPORT_GUARD_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker)
    files = receipt.get("files")
    if receipt.get("signature") != signature or not isinstance(files, dict) or not files:
        raise ValueError("Malformed support guard completion receipt")
    for name, digest in files.items():
        path = Path(out) / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Support guard completion integrity failure: {path}")
    return True


def validate_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-support-guard-v1":
        raise ValueError("Wrong support guard version")
    if config.get("seed") != 43 or config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("Support guard is restricted to seed43 folds")
    if config.get("new_classifier_fits") != 0 or config.get("fixed_weights") != {"h1": 0.5, "c10": 0.5}:
        raise ValueError("Support guard permits no fit and only fixed 50:50 mean")
    if config.get("symbolic_rule") != "all original prepared columns except global|":
        raise ValueError("Support guard symbolic input rule changed")
    if config.get("adaptive_rule_designed_after_error_inspection") is not True:
        raise ValueError("Support guard adaptive provenance must be explicit")
    if any(config.get(key) is not False for key in
           ("test_read", "server_score_used", "submission", "calibration", "threshold_fit")):
        raise ValueError("Forbidden data flow or learned postprocessing enabled")


def segment_report(y, probability, mask, class_order):
    probability = _probability(probability)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(y),):
        raise ValueError("Segment report mask shape mismatch")
    prediction = probability.argmax(1)
    epsilon = float(np.finfo(np.float64).eps)
    result = {"all_rows": len(y), "guarded_rows": int(mask.sum()),
              "nll_clip_epsilon": epsilon, "subset_used_for_promotion": False}
    for name, selected in (("guarded", mask), ("unguarded", ~mask)):
        selected_y = y[selected]
        selected_prediction = prediction[selected]
        selected_probability = probability[selected]
        result[name] = {
            "rows": int(selected.sum()),
            "correct": int((selected_prediction == selected_y).sum()),
            "accuracy": (float((selected_prediction == selected_y).mean()) if len(selected_y) else None),
            "mean_nll": (float(-np.log(np.clip(
                selected_probability[np.arange(len(selected_y)), selected_y], epsilon, 1)).mean())
                         if len(selected_y) else None),
            "predicted_counts": {str(label): int((selected_prediction == index).sum())
                                 for index, label in enumerate(class_order)},
            "true_counts": {str(label): int((selected_y == index).sum())
                            for index, label in enumerate(class_order)},
        }
    return result


def run(config_path: Path, prepared_root: Path, h1_oof: Path, c10_oof: Path,
        h1_run: Path, tfidf_run: Path, out: Path) -> None:
    import lab_metrics as L

    config = T.read_json(config_path)
    validate_config(config)
    helpers = {"tree_helper_sha256": Path(T.__file__),
               "tfidf_helper_sha256": Path(F.__file__)}
    for key, path in helpers.items():
        if T.sha(path) != config.get(key):
            raise ValueError(f"Pinned support guard helper mismatch: {key}")
    if T.sha(h1_oof) != config["h1_oof_sha256"] or T.sha(c10_oof) != config["c10_oof_sha256"]:
        raise ValueError("Support guard source OOF hash mismatch")
    if (T.sha(h1_run / "H1_COMPLETE.json") != config["h1_complete_sha256"] or
            T.sha(tfidf_run / "TFIDF_LOGISTIC_COMPLETE.json") != config["tfidf_complete_sha256"]):
        raise ValueError("Support guard source completion hash mismatch")
    h1 = L.load_oof(h1_oof)
    c10 = L.load_oof(c10_oof)
    validate_exact_alignment(h1, c10)
    signature = T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                          "h1_oof_sha256": T.sha(h1_oof), "c10_oof_sha256": T.sha(c10_oof)})
    if completed_run(out, signature):
        print(json.dumps({"event": "COMPLETED_SUPPORT_GUARD_VERIFIED", "new_fits": 0}), flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Support guard output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    T.write_json(out / "CONTRACT.json", {
        "signature": signature, "config": config, "runner_sha256": T.sha(Path(__file__)),
        "scope": "ADAPTIVE rule designed after error inspection; not independent validation",
        "new_classifier_fits": 0, "test_read": False,
    })
    guard = np.zeros(len(h1["ids"]), dtype=bool)
    fold_evidence = {}
    coverage = np.zeros(len(h1["ids"]), dtype=np.int8)
    for fold in config["folds"]:
        prepared = prepared_root / f"fold_{fold}"
        expected = config["prepared_receipts"][str(fold)]
        if T.sha(prepared / "COMPLETE.json") != expected:
            raise ValueError(f"Support guard prepared receipt changed: fold {fold}")
        T.verify_complete_receipt(prepared)
        train_matrix = sparse.load_npz(prepared / "train.npz").tocsr()
        valid_matrix = sparse.load_npz(prepared / "valid.npz").tocsr()
        with gzip.open(prepared / "encoder.json.gz", "rt", encoding="utf-8") as stream:
            encoder = json.load(stream)
        review = T.read_json(prepared / "review.json")
        with np.load(prepared / "identity.npz", allow_pickle=False) as identity:
            train_indices = identity["train_indices"]
            valid_indices = identity["valid_indices"]
        T.validate_identity(h1["folds"], h1["groups"], fold, train_indices, valid_indices)
        F.verify_encoder_contract(encoder, review, h1["ids"][train_indices].tolist(),
                                  config["representation"])
        names = encoder["names"]
        if train_matrix.shape[1] != len(names) or valid_matrix.shape[1] != len(names):
            raise ValueError(f"Support guard prepared feature order mismatch: fold {fold}")
        take = F.symbolic_indices(names)
        if len(take) != len(names) - 3:
            raise ValueError(f"Support guard must exclude exactly global3: fold {fold}")
        fold_guard, receipt = support_guard_mask(train_matrix[:, take], valid_matrix[:, take])
        guard[valid_indices] = fold_guard
        coverage[valid_indices] += 1
        fold_evidence[str(fold)] = {
            **receipt, "prepared_receipt_sha256": expected,
            "symbolic_names_sha256": T.objsha([names[index] for index in take]),
            "train_ids_sha256": T.objsha(h1["ids"][train_indices].tolist()),
            "valid_ids_sha256": T.objsha(h1["ids"][valid_indices].tolist()),
            "guarded_row_positions_sha256": T.objsha(valid_indices[fold_guard].tolist()),
        }
    if not np.all(coverage == 1):
        raise ValueError("Support guard OOF coverage is not exactly once")
    fixed_mean = guarded_fixed_mean(h1["prob"], c10["prob"], np.zeros(len(guard), dtype=bool))
    guarded = guarded_fixed_mean(h1["prob"], c10["prob"], guard)
    for probability in (fixed_mean, guarded):
        F.validate_oof_probability(probability, len(h1["ids"]), len(h1["class_order"]))
    endpoints = {"h1": h1["prob"], "fixed_mean_50_50": fixed_mean,
                 "support_guarded_fixed_mean": guarded}
    scores = {name: L.metrics(h1, probability) for name, probability in endpoints.items()}
    rows = []
    for name, score in scores.items():
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])}})
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        L.save_npz(out / f"{name}_oof.npz", **{key: h1[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")}, prob=probability)
    pd.DataFrame(rows).to_csv(out / "support_guard_scoreboard.csv", index=False)
    comparisons = {}
    for candidate, baseline in (("fixed_mean_50_50", "h1"),
                                ("support_guarded_fixed_mean", "h1"),
                                ("support_guarded_fixed_mean", "fixed_mean_50_50")):
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "delta_macro_f1": scores[candidate]["macro_f1"] - scores[baseline]["macro_f1"],
            **L.changes(h1["y"], endpoints[candidate], endpoints[baseline]),
            "bootstrap": L.bootstrap(h1, endpoints[candidate], endpoints[baseline], 1000, 43),
        }
    segments = {name: segment_report(h1["y"], probability, guard, h1["class_order"])
                for name, probability in endpoints.items()}
    T.write_json(out / "FOLD_EVIDENCE.json", fold_evidence)
    T.write_json(out / "SEGMENT_DIAGNOSTICS.json", segments)
    T.save_npz(out / "guard_mask.npz", ids=h1["ids"], folds=h1["folds"], guard=guard)
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_SUPPORT_GUARD_COMPLETE",
        "adaptive_rule_designed_after_error_inspection": True,
        "independent_validation": False, "scores": {
            name: {key: value for key, value in score.items() if key != "per_class"}
            for name, score in scores.items()},
        "comparisons": comparisons, "guarded_rows": int(guard.sum()),
        "all_rows_retained": len(guard), "new_classifier_fits": 0,
        "rule_inputs": "train/valid symbolic numeric values only; no labels or IDs",
        "subset_used_for_promotion": False, "automatic_promotion": False,
        "test_read": False, "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "SUPPORT_GUARD_REVIEW.json", review)
    files = ["CONTRACT.json", "SUPPORT_GUARD_REVIEW.json", "support_guard_scoreboard.csv",
             "FOLD_EVIDENCE.json", "SEGMENT_DIAGNOSTICS.json", "guard_mask.npz"]
    for name in endpoints:
        files += [f"{name}_oof.npz", f"{name}_per_class.csv"]
    T.write_json(out / "SUPPORT_GUARD_COMPLETE.json", {
        "signature": signature, "files": {name: T.sha(out / name) for name in files},
        "new_classifier_fits": 0,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--h1-oof", required=True)
    parser.add_argument("--c10-oof", required=True)
    parser.add_argument("--h1-run", required=True)
    parser.add_argument("--tfidf-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        run(Path(args.config).resolve(), Path(args.prepared_root).resolve(),
            Path(args.h1_oof).resolve(), Path(args.c10_oof).resolve(),
            Path(args.h1_run).resolve(), Path(args.tfidf_run).resolve(),
            Path(args.out).resolve())
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        T.write_json(Path(args.out) / "FAILED.json", {
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
