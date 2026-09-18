"""Packaging-corrected v2 of the no-fit support-guarded fixed mean."""
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

import lab_metrics as L
import goal055_tree_budget as T
import goal055_tfidf_logistic as F
import goal055_support_guard as V1


def save_endpoint_oofs(out: Path, reference: dict, endpoints: dict[str, np.ndarray]):
    out.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name, endpoint_probability in endpoints.items():
        endpoint_probability = np.asarray(endpoint_probability, dtype=np.float64)
        F.validate_oof_probability(endpoint_probability, len(reference["ids"]),
                                   len(reference["class_order"]))
        path = out / f"{name}_oof.npz"
        L.save_npz(path, **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")},
                   prob=endpoint_probability)
        with np.load(path, allow_pickle=False) as raw_stored:
            stored_raw_probability = np.asarray(raw_stored["prob"], dtype=np.float64)
        raw_maximum = float(np.max(np.abs(stored_raw_probability - endpoint_probability)))
        if not np.array_equal(stored_raw_probability, endpoint_probability):
            raise ValueError(f"Raw endpoint OOF readback mismatch: {name}")
        stored = L.load_oof(path)
        V1.validate_exact_alignment(reference, stored)
        normalized_maximum = float(np.max(np.abs(stored["prob"] - endpoint_probability)))
        source_score = L.metrics(reference, endpoint_probability)["macro_f1"]
        stored_score = L.metrics(reference, stored["prob"])["macro_f1"]
        if normalized_maximum > 1e-12 or stored_score != source_score:
            raise ValueError(f"Endpoint OOF readback mismatch: {name}")
        receipt[name] = {
            "oof_sha256": T.sha(path), "max_source_probability_abs": raw_maximum,
            "normalized_readback_max_abs": normalized_maximum,
            "normalized_readback_tolerance": 1e-12,
            "source_macro_f1": source_score, "stored_macro_f1": stored_score,
            "rows": len(reference["ids"]), "classes": len(reference["class_order"]),
        }
    return receipt


def validate_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-support-guard-v2":
        raise ValueError("Wrong support guard v2 version")
    if config.get("seed") != 43 or config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("Support guard v2 is restricted to seed43 folds")
    if config.get("new_classifier_fits") != 0 or config.get("fixed_weights") != {"h1": 0.5, "c10": 0.5}:
        raise ValueError("Support guard v2 permits no fit and only fixed 50:50 mean")
    if config.get("symbolic_rule") != "all original prepared columns except global|":
        raise ValueError("Support guard v2 symbolic rule changed")
    if config.get("adaptive_rule_designed_after_error_inspection") is not True:
        raise ValueError("Support guard v2 adaptive provenance must be explicit")
    if any(config.get(key) is not False for key in
           ("test_read", "server_score_used", "submission", "calibration", "threshold_fit")):
        raise ValueError("Forbidden data flow or learned postprocessing enabled")


def completed_run(out: Path, signature: str) -> bool:
    marker = out / "SUPPORT_GUARD_V2_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker)
    files = receipt.get("files")
    if receipt.get("signature") != signature or not isinstance(files, dict) or not files:
        raise ValueError("Malformed support guard v2 completion receipt")
    for name, digest in files.items():
        path = out / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Support guard v2 completion integrity failure: {path}")
    return True


def run(config_path: Path, prepared_root: Path, h1_oof: Path, c10_oof: Path,
        h1_run: Path, tfidf_run: Path, out: Path) -> None:
    config = T.read_json(config_path)
    validate_config(config)
    helpers = {"tree_helper_sha256": Path(T.__file__),
               "tfidf_helper_sha256": Path(F.__file__),
               "v1_helper_sha256": Path(V1.__file__)}
    for key, path in helpers.items():
        if T.sha(path) != config.get(key):
            raise ValueError(f"Pinned support guard v2 helper mismatch: {key}")
    if T.sha(h1_oof) != config["h1_oof_sha256"] or T.sha(c10_oof) != config["c10_oof_sha256"]:
        raise ValueError("Support guard v2 source OOF hash mismatch")
    if (T.sha(h1_run / "H1_COMPLETE.json") != config["h1_complete_sha256"] or
            T.sha(tfidf_run / "TFIDF_LOGISTIC_COMPLETE.json") != config["tfidf_complete_sha256"]):
        raise ValueError("Support guard v2 source completion hash mismatch")
    h1 = L.load_oof(h1_oof)
    c10 = L.load_oof(c10_oof)
    V1.validate_exact_alignment(h1, c10)
    signature = T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                          "h1_oof_sha256": T.sha(h1_oof), "c10_oof_sha256": T.sha(c10_oof)})
    if completed_run(out, signature):
        print(json.dumps({"event": "COMPLETED_SUPPORT_GUARD_V2_VERIFIED", "new_fits": 0}), flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Support guard v2 output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    T.write_json(out / "CONTRACT.json", {
        "signature": signature, "config": config, "runner_sha256": T.sha(Path(__file__)),
        "scope": "ADAPTIVE rule designed after error inspection; packaging-only correction from v1",
        "v1_output_used": False, "new_classifier_fits": 0, "test_read": False,
    })
    guard = np.zeros(len(h1["ids"]), dtype=bool)
    coverage = np.zeros(len(h1["ids"]), dtype=np.int8)
    fold_evidence = {}
    for fold in config["folds"]:
        prepared = prepared_root / f"fold_{fold}"
        expected = config["prepared_receipts"][str(fold)]
        if T.sha(prepared / "COMPLETE.json") != expected:
            raise ValueError(f"Support guard v2 prepared receipt changed: fold {fold}")
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
            raise ValueError(f"Support guard v2 prepared feature order mismatch: fold {fold}")
        take = F.symbolic_indices(names)
        if len(take) != len(names) - 3:
            raise ValueError(f"Support guard v2 must exclude exactly global3: fold {fold}")
        fold_guard, rule_receipt = V1.support_guard_mask(
            train_matrix[:, take], valid_matrix[:, take])
        guard[valid_indices] = fold_guard
        coverage[valid_indices] += 1
        fold_evidence[str(fold)] = {
            **rule_receipt, "prepared_receipt_sha256": expected,
            "symbolic_names_sha256": T.objsha([names[index] for index in take]),
            "train_ids_sha256": T.objsha(h1["ids"][train_indices].tolist()),
            "valid_ids_sha256": T.objsha(h1["ids"][valid_indices].tolist()),
            "guarded_row_positions_sha256": T.objsha(valid_indices[fold_guard].tolist()),
        }
    if not np.all(coverage == 1):
        raise ValueError("Support guard v2 OOF coverage is not exactly once")
    fixed_mean = V1.guarded_fixed_mean(h1["prob"], c10["prob"], np.zeros(len(guard), bool))
    guarded = V1.guarded_fixed_mean(h1["prob"], c10["prob"], guard)
    endpoints = {"h1": h1["prob"], "fixed_mean_50_50": fixed_mean,
                 "support_guarded_fixed_mean": guarded}
    scores = {name: L.metrics(h1, probability) for name, probability in endpoints.items()}
    readback = save_endpoint_oofs(out, h1, endpoints)
    for name, score in scores.items():
        if readback[name]["stored_macro_f1"] != score["macro_f1"]:
            raise ValueError(f"Score/OOF concordance failure: {name}")
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
    rows = [{"endpoint": name,
             **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
             **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])} }
            for name, score in scores.items()]
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
    segments = {name: V1.segment_report(h1["y"], probability, guard, h1["class_order"])
                for name, probability in endpoints.items()}
    T.write_json(out / "OOF_READBACK.json", {
        "source_h1_oof_sha256": T.sha(h1_oof),
        "source_c10_oof_sha256": T.sha(c10_oof),
        "endpoints": readback,
    })
    T.write_json(out / "FOLD_EVIDENCE.json", fold_evidence)
    T.write_json(out / "SEGMENT_DIAGNOSTICS.json", segments)
    T.save_npz(out / "guard_mask.npz", ids=h1["ids"], folds=h1["folds"], guard=guard)
    review = {
        "status": "ADAPTIVE_DEVELOPMENT_SUPPORT_GUARD_V2_COMPLETE",
        "adaptive_rule_designed_after_error_inspection": True,
        "independent_validation": False,
        "scores": {name: {key: value for key, value in score.items() if key != "per_class"}
                   for name, score in scores.items()},
        "comparisons": comparisons, "guarded_rows": int(guard.sum()),
        "all_rows_retained": len(guard), "new_classifier_fits": 0,
        "packaging_fix_only_from_v1": True,
        "source_oof_readback_verified": True,
        "rule_inputs": "train/valid symbolic numeric values only; no labels or IDs",
        "subset_used_for_promotion": False, "automatic_promotion": False,
        "test_read": False, "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "SUPPORT_GUARD_V2_REVIEW.json", review)
    files = ["CONTRACT.json", "SUPPORT_GUARD_V2_REVIEW.json", "support_guard_scoreboard.csv",
             "OOF_READBACK.json", "FOLD_EVIDENCE.json", "SEGMENT_DIAGNOSTICS.json",
             "guard_mask.npz"]
    for name in endpoints:
        files += [f"{name}_oof.npz", f"{name}_per_class.csv"]
    T.write_json(out / "SUPPORT_GUARD_V2_COMPLETE.json", {
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
