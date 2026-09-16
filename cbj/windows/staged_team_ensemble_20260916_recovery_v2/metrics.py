"""Terminal-only exact OOF assembly, readback and Macro-F1 scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from base_receipts import sequence_sha256


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble_final_oof(fold_results, ids, y, groups, folds, class_order, path):
    ids, y, groups, folds = map(np.asarray, (ids, y, groups, folds))
    if any(value.shape != ids.shape for value in (y, groups, folds)):
        raise ValueError("final OOF identity arrays are misaligned")
    rows, classes = len(ids), len(class_order)
    group_fold = {}
    for group, fold in zip(groups.tolist(), folds.tolist()):
        previous = group_fold.setdefault(str(group), int(fold))
        if previous != int(fold):
            raise ValueError("canonical group crosses outer folds")
    probability = np.zeros((rows, classes), dtype=np.float64)
    coverage = np.zeros(rows, dtype=np.uint8)
    seen_folds = set()
    fold_receipts = []
    for result in fold_results:
        fold = int(result["fold"])
        indices = np.asarray(result["valid_indices"], dtype=np.int64)
        values = np.asarray(result["probability"], dtype=np.float64)
        if fold in seen_folds or values.shape != (len(indices), classes):
            raise ValueError("outer fold result is duplicated or malformed")
        if np.any(indices < 0) or np.any(indices >= rows) or not np.all(folds[indices] == fold):
            raise ValueError("outer fold identity mismatch")
        seen_folds.add(fold)
        probability[indices] = values
        coverage[indices] += 1
        fold_receipts.append({
            "fold": fold,
            "valid_indices_sha256": sequence_sha256(indices),
            "valid_ids_sha256": sequence_sha256(ids[indices]),
            "probability_sha256": hashlib.sha256(
                np.ascontiguousarray(values, dtype="<f8").tobytes()
            ).hexdigest(),
        })
    if not np.all(coverage == 1):
        raise ValueError("final OOF coverage is not exactly once")
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=1e-8):
        raise ValueError("final OOF probability is invalid")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, ids=ids, y=y, groups=groups, folds=folds,
        class_order=np.asarray(class_order), probability=probability, coverage=coverage,
    )
    receipt = {
        "schema_version": "FINAL_PROCEDURE_OOF_RECEIPT_V1",
        "oof_sha256": _sha_file(path),
        "rows": rows,
        "classes": classes,
        "ids_sha256": sequence_sha256(ids),
        "y_sha256": sequence_sha256(y),
        "groups_sha256": sequence_sha256(groups),
        "folds_sha256": sequence_sha256(folds),
        "coverage_sha256": sequence_sha256(coverage),
        "fold_receipts": sorted(fold_receipts, key=lambda row: row["fold"]),
        "scored_after_complete_readback": False,
        "test_read": False,
    }
    return receipt


def load_final_oof(path, receipt):
    path = Path(path)
    if receipt.get("schema_version") != "FINAL_PROCEDURE_OOF_RECEIPT_V1" or receipt.get("oof_sha256") != _sha_file(path):
        raise ValueError("final OOF receipt hash mismatch")
    with np.load(path, allow_pickle=False) as stored:
        result = {key: np.asarray(stored[key]) for key in stored.files}
    for key in ("ids", "y", "groups", "folds", "coverage"):
        expected = receipt.get(f"{key}_sha256")
        if sequence_sha256(result[key]) != expected:
            raise ValueError(f"final OOF {key} readback mismatch")
    if result["probability"].shape != (receipt.get("rows"), receipt.get("classes")):
        raise ValueError("final OOF probability readback shape mismatch")
    return result


def score_final_oof(oof):
    coverage = np.asarray(oof["coverage"])
    probability = np.asarray(oof["probability"], dtype=np.float64)
    y = np.asarray(oof["y"])
    classes = len(oof["class_order"])
    if coverage.shape != (len(y),) or not np.all(coverage == 1):
        raise ValueError("cannot score incomplete final OOF")
    score = float(f1_score(
        y, probability.argmax(axis=1), labels=np.arange(classes),
        average="macro", zero_division=0,
    ))
    return {
        "schema_version": "FINAL_PROCEDURE_SCORE_V1",
        "macro_f1": score,
        "rows": len(y),
        "classes": classes,
        "terminal_only": True,
        "used_for_grid_or_stopping": False,
    }


def milestone_status(macro_f1):
    macro_f1 = float(macro_f1)
    if not np.isfinite(macro_f1) or macro_f1 < 0 or macro_f1 > 1:
        raise ValueError("milestone score is invalid")
    return {
        "schema_version": "TEAM_BUNDLE_MILESTONE_V1",
        "status": "TARGET_REACHED" if macro_f1 >= 0.60 else "BUNDLE_COMPLETE_REVIEW",
        "macro_f1": macro_f1,
        "target": 0.60,
        "automatic_next_bundle": False,
        "submission": False,
    }
