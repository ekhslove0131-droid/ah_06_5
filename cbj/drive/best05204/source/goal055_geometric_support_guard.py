"""No-fit fixed geometric H1/C10 pool with the frozen input-only support guard."""
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


def validate_probability(probability, rows: int, classes: int) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if (probability.shape != (rows, classes) or not np.isfinite(probability).all() or
            (probability < 0).any() or
            not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise ValueError("Geometric support-guard probability is invalid")
    return probability


def fixed_geometric_pool(h1_probability, c10_probability, epsilon: float = 1e-300):
    h1 = np.asarray(h1_probability, dtype=np.float64)
    c10 = np.asarray(c10_probability, dtype=np.float64)
    if h1.shape != c10.shape or h1.ndim != 2 or epsilon != 1e-300:
        raise ValueError("Geometric pool shape or fixed epsilon changed")
    if (not np.isfinite(h1).all() or not np.isfinite(c10).all() or
            (h1 < 0).any() or (c10 < 0).any() or
            not np.allclose(h1.sum(axis=1), 1.0, atol=1e-12, rtol=0) or
            not np.allclose(c10.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise ValueError("Geometric pool source probability is invalid")
    log_score = 0.5 * np.log(np.maximum(h1, epsilon)) + 0.5 * np.log(np.maximum(c10, epsilon))
    row_max = np.max(log_score, axis=1, keepdims=True)
    unnormalized = np.exp(log_score - row_max)
    denominator = unnormalized.sum(axis=1, keepdims=True)
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("Geometric pool normalization failed")
    return unnormalized / denominator


def guarded_geometric_pool(h1_probability, c10_probability, guard_mask,
                           epsilon: float = 1e-300):
    h1 = np.asarray(h1_probability, dtype=np.float64)
    guard = np.asarray(guard_mask, dtype=bool)
    if guard.shape != (h1.shape[0],):
        raise ValueError("Geometric support guard mask shape mismatch")
    result = fixed_geometric_pool(h1, c10_probability, epsilon)
    result[guard] = h1[guard]
    if not np.array_equal(result[guard], h1[guard]):
        raise ValueError("Geometric guard did not preserve exact H1 fallback")
    return result


def validate_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-geometric-support-guard-v1":
        raise ValueError("Wrong geometric support-guard version")
    if config.get("seed") != 43 or config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("Geometric support guard is restricted to seed43 folds")
    if config.get("new_classifier_fits") != 0 or config.get("fixed_weights") != {"h1": 0.5, "c10": 0.5}:
        raise ValueError("Geometric support guard permits no fit and fixed equal weights only")
    if config.get("epsilon") != 1e-300 or config.get("pool") != "log-probability geometric mean":
        raise ValueError("Geometric support-guard formula changed")
    if config.get("symbolic_rule") != "all original prepared columns except global|":
        raise ValueError("Geometric support-guard symbolic rule changed")
    if config.get("adaptive_reused_oof_screen") is not True:
        raise ValueError("Geometric support-guard adaptive provenance must be explicit")
    if any(config.get(key) is not False for key in
           ("test_read", "server_score_used", "submission", "calibration",
            "threshold_fit", "weight_search", "epsilon_search")):
        raise ValueError("Forbidden geometric support-guard data flow or search enabled")


def save_endpoint_oofs(out: Path, reference: dict, endpoints: dict[str, np.ndarray]):
    receipt = {}
    for name, endpoint_probability in endpoints.items():
        probability = validate_probability(endpoint_probability, len(reference["ids"]),
                                           len(reference["class_order"]))
        path = Path(out) / f"{name}_oof.npz"
        L.save_npz(path, **{key: reference[key]
                   for key in ("ids", "y", "groups", "folds", "class_order")},
                   prob=probability)
        with np.load(path, allow_pickle=False) as raw:
            if not np.array_equal(raw["prob"], probability):
                raise ValueError(f"Raw geometric endpoint export mismatch: {name}")
            for key in ("ids", "y", "groups", "folds", "class_order"):
                if not np.array_equal(raw[key], reference[key]):
                    raise ValueError(f"Geometric endpoint identity export mismatch: {name}/{key}")
        loaded = L.load_oof(path)
        V1.validate_exact_alignment(reference, loaded)
        normalized_max = float(np.max(np.abs(loaded["prob"] - probability)))
        source_score = L.metrics(reference, probability)["macro_f1"]
        loaded_score = L.metrics(reference, loaded["prob"])["macro_f1"]
        if normalized_max > 1e-12 or source_score != loaded_score:
            raise ValueError(f"Geometric endpoint score/readback mismatch: {name}")
        receipt[name] = {"oof_sha256": T.sha(path), "normalized_max_abs": normalized_max,
                         "source_macro_f1": source_score, "stored_macro_f1": loaded_score}
    if len({item["oof_sha256"] for item in receipt.values()}) != len(receipt):
        raise ValueError("Geometric support-guard endpoints are not distinct artifacts")
    return receipt


def reconstruct_guard(config: dict, prepared_root: Path, reference: dict):
    guard = np.zeros(len(reference["ids"]), dtype=bool)
    coverage = np.zeros(len(reference["ids"]), dtype=np.int8)
    evidence = {}
    for fold in config["folds"]:
        prepared = prepared_root / f"fold_{fold}"
        expected = config["prepared_receipts"][str(fold)]
        if T.sha(prepared / "COMPLETE.json") != expected:
            raise ValueError(f"Geometric guard prepared receipt changed: fold {fold}")
        T.verify_complete_receipt(prepared)
        train_matrix = sparse.load_npz(prepared / "train.npz").tocsr()
        valid_matrix = sparse.load_npz(prepared / "valid.npz").tocsr()
        with gzip.open(prepared / "encoder.json.gz", "rt", encoding="utf-8") as stream:
            encoder = json.load(stream)
        review = T.read_json(prepared / "review.json")
        with np.load(prepared / "identity.npz", allow_pickle=False) as identity:
            train_indices = identity["train_indices"]
            valid_indices = identity["valid_indices"]
        T.validate_identity(reference["folds"], reference["groups"], fold,
                            train_indices, valid_indices)
        F.verify_encoder_contract(encoder, review, reference["ids"][train_indices].tolist(),
                                  config["representation"])
        names = encoder["names"]
        if train_matrix.shape[1] != len(names) or valid_matrix.shape[1] != len(names):
            raise ValueError(f"Geometric guard prepared feature order mismatch: fold {fold}")
        take = F.symbolic_indices(names)
        if len(take) != len(names) - 3:
            raise ValueError(f"Geometric guard must exclude exactly global3: fold {fold}")
        fold_guard, rule_receipt = V1.support_guard_mask(train_matrix[:, take], valid_matrix[:, take])
        guard[valid_indices] = fold_guard
        coverage[valid_indices] += 1
        evidence[str(fold)] = {
            **rule_receipt, "prepared_receipt_sha256": expected,
            "symbolic_names_sha256": T.objsha([names[index] for index in take]),
            "train_ids_sha256": T.objsha(reference["ids"][train_indices].tolist()),
            "valid_ids_sha256": T.objsha(reference["ids"][valid_indices].tolist()),
            "guarded_row_positions_sha256": T.objsha(valid_indices[fold_guard].tolist()),
            "labels_or_ids_used_by_mask_function": False,
        }
    if not np.all(coverage == 1) or int(guard.sum()) != config["expected_guarded_rows"]:
        raise ValueError("Geometric support guard coverage/count mismatch")
    return guard, evidence


def expected_root_files():
    endpoints = ("h1", "arithmetic_support_guard", "geometric_support_guard")
    files = ["CONTRACT.json", "GEOMETRIC_SUPPORT_GUARD_REVIEW.json",
             "geometric_support_guard_scoreboard.csv",
             "OOF_READBACK.json", "SOURCE_REPLAY.json", "FOLD_EVIDENCE.json",
             "FORMULA.json", "guard_mask.npz"]
    for name in endpoints:
        files += [f"{name}_oof.npz", f"{name}_per_class.csv"]
    return files


def run_signature(config: dict, h1_oof: Path, c10_oof: Path, arithmetic_oof: Path):
    return T.objsha({"config": config, "runner_sha256": T.sha(Path(__file__)),
                     "h1_oof_sha256": T.sha(h1_oof), "c10_oof_sha256": T.sha(c10_oof),
                     "arithmetic_oof_sha256": T.sha(arithmetic_oof)})


def completed_run(out: Path, signature: str) -> bool:
    marker = out / "GEOMETRIC_SUPPORT_GUARD_COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = T.read_json(marker); files = receipt.get("files")
    if (receipt.get("signature") != signature or not isinstance(files, dict) or
            set(files) != set(expected_root_files()) or receipt.get("new_classifier_fits") != 0):
        raise ValueError("Malformed geometric support-guard completion receipt")
    for name, digest in files.items():
        path = out / name
        if not path.is_file() or T.sha(path) != digest:
            raise ValueError(f"Geometric support-guard completion integrity failure: {path}")
    return True


def run(config_path: Path, prepared_root: Path, h1_oof: Path, c10_oof: Path,
        arithmetic_oof: Path, h1_run: Path, tfidf_run: Path, arithmetic_run: Path,
        out: Path) -> None:
    config = T.read_json(config_path)
    validate_config(config)
    helpers = {"tree_helper_sha256": Path(T.__file__),
               "tfidf_helper_sha256": Path(F.__file__),
               "v1_helper_sha256": Path(V1.__file__),
               "lab_metrics_sha256": Path(L.__file__)}
    for key, path in helpers.items():
        if T.sha(path) != config.get(key):
            raise ValueError(f"Pinned geometric support-guard helper mismatch: {key}")
    source_paths = {"h1_oof_sha256": h1_oof, "c10_oof_sha256": c10_oof,
                    "arithmetic_oof_sha256": arithmetic_oof,
                    "h1_complete_sha256": h1_run / "H1_COMPLETE.json",
                    "tfidf_complete_sha256": tfidf_run / "TFIDF_LOGISTIC_COMPLETE.json",
                    "arithmetic_complete_sha256": arithmetic_run / "SUPPORT_GUARD_V2_COMPLETE.json"}
    for key, path in source_paths.items():
        if T.sha(path) != config[key]:
            raise ValueError(f"Geometric support-guard source hash mismatch: {key}")
    h1 = L.load_oof(h1_oof); c10 = L.load_oof(c10_oof); arithmetic = L.load_oof(arithmetic_oof)
    V1.validate_exact_alignment(h1, c10); V1.validate_exact_alignment(h1, arithmetic)
    signature = run_signature(config, h1_oof, c10_oof, arithmetic_oof)
    if completed_run(out, signature):
        print(json.dumps({"event": "COMPLETED_GEOMETRIC_SUPPORT_GUARD_VERIFIED",
                          "new_fits": 0}), flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Geometric support-guard output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    T.write_json(out / "CONTRACT.json", {
        "signature": signature, "config": config, "runner_sha256": T.sha(Path(__file__)),
        "scope": "adaptive reused-OOF fixed equal geometric pool with frozen input-only guard",
        "independent_validation": False, "new_classifier_fits": 0, "test_read": False,
    })
    T.write_json(out / "FORMULA.json", {
        "ordinary_rows": "normalize(exp(0.5*log(max(H1,1e-300))+0.5*log(max(C10,1e-300))))",
        "normalization": "stable per-row log-sum-exp via subtraction of row maximum",
        "guarded_rows": "exact H1 float64 probability",
        "fixed_weights": config["fixed_weights"], "epsilon": config["epsilon"],
        "weight_search": False, "epsilon_search": False, "calibration": False,
    })
    guard, evidence = reconstruct_guard(config, prepared_root, h1)
    arithmetic_rebuilt = V1.guarded_fixed_mean(h1["prob"], c10["prob"], guard)
    arithmetic_max = float(np.max(np.abs(arithmetic_rebuilt - arithmetic["prob"])))
    if (arithmetic_max > config["source_replay_tolerance"] or
            not np.array_equal(arithmetic_rebuilt.argmax(1), arithmetic["prob"].argmax(1)) or
            L.metrics(h1, arithmetic_rebuilt)["macro_f1"] != L.metrics(h1, arithmetic["prob"])["macro_f1"]):
        raise ValueError("Frozen arithmetic support guard did not replay")
    geometric = guarded_geometric_pool(h1["prob"], c10["prob"], guard, config["epsilon"])
    endpoints = {"h1": h1["prob"], "arithmetic_support_guard": arithmetic["prob"],
                 "geometric_support_guard": geometric}
    scores = {name: L.metrics(h1, probability) for name, probability in endpoints.items()}
    readback = save_endpoint_oofs(out, h1, endpoints)
    rows = []
    for name, score in scores.items():
        if readback[name]["stored_macro_f1"] != score["macro_f1"]:
            raise ValueError(f"Geometric support-guard score concordance failed: {name}")
        pd.DataFrame(score["per_class"]).to_csv(out / f"{name}_per_class.csv", index=False)
        rows.append({"endpoint": name,
                     **{key: value for key, value in score.items() if key not in ("per_class", "fold_f1")},
                     **{f"fold_{fold}": value for fold, value in enumerate(score["fold_f1"])}})
    pd.DataFrame(rows).to_csv(out / "geometric_support_guard_scoreboard.csv", index=False)
    comparisons = {}
    for baseline in ("h1", "arithmetic_support_guard"):
        comparisons[f"geometric_support_guard_vs_{baseline}"] = {
            "delta_macro_f1": scores["geometric_support_guard"]["macro_f1"] - scores[baseline]["macro_f1"],
            **L.changes(h1["y"], geometric, endpoints[baseline]),
            "bootstrap": L.bootstrap(h1, geometric, endpoints[baseline],
                                     config["bootstrap_resamples"], 43),
        }
    T.write_json(out / "SOURCE_REPLAY.json", {
        "source_h1_oof_sha256": config["h1_oof_sha256"],
        "source_c10_oof_sha256": config["c10_oof_sha256"],
        "source_arithmetic_oof_sha256": config["arithmetic_oof_sha256"],
        "arithmetic_replay_max_abs": arithmetic_max,
        "arithmetic_argmax_exact": True, "arithmetic_macro_f1_exact": True,
    })
    T.write_json(out / "FOLD_EVIDENCE.json", evidence)
    T.save_npz(out / "guard_mask.npz", ids=h1["ids"], folds=h1["folds"], guard=guard)
    T.write_json(out / "OOF_READBACK.json", {"endpoints": readback})
    review = {
        "status": "ADAPTIVE_REUSED_OOF_GEOMETRIC_SUPPORT_GUARD_COMPLETE",
        "adaptive_reused_oof_screen": True, "independent_validation": False,
        "scores": {name: {key: value for key, value in score.items() if key != "per_class"}
                   for name, score in scores.items()},
        "comparisons": comparisons, "guarded_rows": int(guard.sum()),
        "all_rows_retained": len(guard), "new_classifier_fits": 0,
        "rule_inputs": "outer-train/heldout symbolic numeric support only; no labels or IDs",
        "fixed_pool_weights": config["fixed_weights"], "automatic_promotion": False,
        "test_read": False, "server_score_used": False, "submission_created": False,
    }
    T.write_json(out / "GEOMETRIC_SUPPORT_GUARD_REVIEW.json", review)
    T.write_json(out / "GEOMETRIC_SUPPORT_GUARD_COMPLETE.json", {
        "signature": signature, "files": {name: T.sha(out / name) for name in expected_root_files()},
        "new_classifier_fits": 0})
    if not completed_run(out, signature):
        raise ValueError("Geometric support-guard final completion verification failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--h1-oof", required=True)
    parser.add_argument("--c10-oof", required=True)
    parser.add_argument("--arithmetic-oof", required=True)
    parser.add_argument("--h1-run", required=True)
    parser.add_argument("--tfidf-run", required=True)
    parser.add_argument("--arithmetic-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        run(Path(args.config).resolve(), Path(args.prepared_root).resolve(),
            Path(args.h1_oof).resolve(), Path(args.c10_oof).resolve(),
            Path(args.arithmetic_oof).resolve(), Path(args.h1_run).resolve(),
            Path(args.tfidf_run).resolve(), Path(args.arithmetic_run).resolve(),
            Path(args.out).resolve())
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        T.write_json(Path(args.out) / "FAILED.json", {"error": str(exc),
                     "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
