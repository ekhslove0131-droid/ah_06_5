"""v97: seal and majority-ensemble three prevalidated v96 group splits."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "trial" / "실험"
DATA = ROOT / "dataset"
SUBMISSION = ROOT / "submission"
HERE = Path(__file__).resolve().parent
OUT = HERE / "validation_outputs_v97_sealed_gene_router_ensemble"
BASE = EXP / "kaggle_고급전략_v96_cbj_gene_aware_decisive_selector"
SETTING = "g512_c0.08_d8_l6_m0.55"
SEEDS = [0, 9617, 9631]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def majority(parent, matrix):
    result = parent.copy()
    for row in range(len(parent)):
        values, counts = np.unique(matrix[row], return_counts=True)
        winner = values[counts.argmax()]
        if counts.max() >= 2:
            result[row] = winner
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decisions, oofs, tests = [], [], []
    for seed in SEEDS:
        folder = BASE / f"validation_outputs_v96_cbj_gene_aware_decisive_selector_{SETTING}_f{seed}"
        decision = json.loads((folder / "v96_decision.json").read_text(encoding="utf-8"))
        decisions.append(decision)
        oofs.append(pd.read_csv(folder / "v96_oof_labels.csv"))
        tests.append(pd.read_csv(folder / "v96_test_labels.csv"))
    key = ["ID", "true_label", "parent_label"]
    for frame in oofs[1:]:
        if not oofs[0][key].equals(frame[key]):
            raise ValueError("OOF identity differs across sealed splits")
    for frame in tests[1:]:
        if not tests[0][["ID", "parent_label"]].equals(frame[["ID", "parent_label"]]):
            raise ValueError("test identity differs across sealed splits")
    classes = np.asarray(sorted(oofs[0].true_label.unique()))
    index = {name: i for i, name in enumerate(classes)}
    y = oofs[0].true_label.map(index).to_numpy()
    parent = oofs[0].parent_label.map(index).to_numpy()
    parent_test = tests[0].parent_label.map(index).to_numpy()
    oof_matrix = np.column_stack([
        frame.candidate_label.map(index).to_numpy() for frame in oofs])
    test_matrix = np.column_stack([
        frame.candidate_label.map(index).to_numpy() for frame in tests])
    candidate = majority(parent, oof_matrix)
    candidate_test = majority(parent_test, test_matrix)
    parent_score = f1_score(y, parent, average="macro", zero_division=0)
    candidate_score = f1_score(y, candidate, average="macro", zero_division=0)
    canonical_folds = oofs[0].fold.to_numpy(int)
    fold_rows = []
    for fold in sorted(np.unique(canonical_folds)):
        mask = canonical_folds == fold
        before = f1_score(y[mask], parent[mask], average="macro", zero_division=0)
        after = f1_score(y[mask], candidate[mask], average="macro", zero_division=0)
        fold_rows.append({"fold": int(fold), "baseline_macro_f1": before,
                          "candidate_macro_f1": after, "delta": after - before})
    fold_frame = pd.DataFrame(fold_rows)
    class_rows = []
    for i, name in enumerate(classes):
        before = f1_score(y == i, parent == i, zero_division=0)
        after = f1_score(y == i, candidate == i, zero_division=0)
        class_rows.append({"class": name, "support": int((y == i).sum()),
                           "baseline_f1": before, "candidate_f1": after,
                           "delta": after - before})
    class_frame = pd.DataFrame(class_rows)
    changes = candidate_test != parent_test
    source_audit_passed = all(
        d["oof_delta"] >= 0.005 and d["positive_folds"] == 5
        and d["worst_fold_delta"] > 0 for d in decisions)
    qa_passed = bool(
        source_audit_passed and candidate_score - parent_score >= 0.005
        and (fold_frame.delta > 0).sum() == 5 and fold_frame.delta.min() > 0
        and class_frame.delta.min() >= -0.015
        and 20 <= changes.sum() <= 80
    )
    decision = {
        "candidate_id": "v97_sealed_gene_router_ensemble",
        "source_split_results": [{
            "fold_seed": d["selector_parameters"]["fold_seed"],
            "oof_macro_f1": d["candidate_oof_macro_f1"],
            "delta": d["oof_delta"], "positive_folds": d["positive_folds"],
            "worst_fold_delta": d["worst_fold_delta"],
            "worst_class_delta": d["worst_class_delta"]} for d in decisions],
        "parent_oof_macro_f1": float(parent_score),
        "ensemble_oof_macro_f1": float(candidate_score),
        "oof_delta": float(candidate_score - parent_score),
        "positive_canonical_folds": int((fold_frame.delta > 0).sum()),
        "worst_canonical_fold_delta": float(fold_frame.delta.min()),
        "worst_class_delta": float(class_frame.delta.min()),
        "oof_label_differences": int(np.sum(candidate != parent)),
        "test_label_differences": int(changes.sum()),
        "test_change_rate": float(changes.mean()),
        "source_audit_passed": source_audit_passed,
        "final_qa_passed": qa_passed,
        "submission_created": False,
    }
    fold_frame.to_csv(OUT / "v97_fold_metrics.csv", index=False)
    class_frame.to_csv(OUT / "v97_class_metrics.csv", index=False)
    transitions = pd.crosstab(
        pd.Series(classes[parent_test[changes]], name="from"),
        pd.Series(classes[candidate_test[changes]], name="to"),
    ).stack().reset_index(name="rows").sort_values("rows", ascending=False)
    transitions.to_csv(OUT / "v97_test_transitions.csv", index=False)
    if qa_passed:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        if not sample.ID.astype(str).equals(tests[0].ID.astype(str)):
            raise ValueError("sample ID order differs")
        sample["SUBCLASS"] = classes[candidate_test]
        path = SUBMISSION / ("submission_v97_sealed_gene_router_ensemble_validated_"
                             + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
        sample.to_csv(path, index=False)
        duplicate = []
        digest = sha(path)
        for other in SUBMISSION.glob("*.csv"):
            if other != path and sha(other) == digest:
                duplicate.append(str(other))
        if duplicate:
            path.unlink()
            raise ValueError(f"duplicate submission: {duplicate}")
        decision.update({"submission_created": True, "submission_path": str(path),
                         "submission_sha256": digest})
    (OUT / "v97_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
