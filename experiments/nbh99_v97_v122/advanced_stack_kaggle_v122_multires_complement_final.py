"""v122: validated multi-resolution complementary-profile correction.

Starting from the proven v97 labels, change a test row only when exact variant
and exact gene-set lookups independently infer the same train-only complement.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "trial" / "실험"
DATA = ROOT / "dataset"
SUBMISSION = ROOT / "submission"
HERE = Path(__file__).resolve().parent
OUT = HERE / "validation_outputs_v122_multires_complement_final"
V92 = EXP / "kaggle_고급전략_v92_cbj_crossfit_metastacker"
V101 = EXP / "kaggle_고급전략_v101_direct_pair_gene_specialists"
V119 = EXP / "kaggle_고급전략_v119_exact_profile_lookup"
sys.path[:0] = [str(V92), str(V101), str(V119)]
import advanced_stack_kaggle_v92_cbj_crossfit_metastacker as v92  # noqa: E402
import advanced_stack_kaggle_v101_direct_pair_gene_specialists as v101  # noqa: E402
import v119_exact_profile_lookup_audit as v119  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(sig, burden, sigt, burdent, y, base):
    groups = defaultdict(list)
    for i, key in enumerate(sig):
        if burden[i] > 0:
            groups[key].append(i)
    directed = defaultdict(Counter)
    pairs = []
    for indices in groups.values():
        if len(indices) == 2 and y[indices[0]] != y[indices[1]]:
            a, b = map(int, y[indices])
            directed[a][b] += 1
            directed[b][a] += 1
            pairs.append(indices)
    mapping, purities, supports = {}, [], []
    for source, counts in directed.items():
        target, support = counts.most_common(1)[0]
        purity = support / sum(counts.values())
        if support >= 20 and purity >= 0.98:
            mapping[source] = target
            purities.append(purity)
            supports.append(support)
    correct, total = 0, 0
    for a, b in pairs:
        if int(y[b]) in mapping:
            correct += int(mapping[int(y[b])] == y[a]); total += 1
        if int(y[a]) in mapping:
            correct += int(mapping[int(y[a])] == y[b]); total += 1
    singleton = {key: int(y[ix[0]]) for key, ix in groups.items() if len(ix) == 1}
    candidate = base.copy()
    matched = 0
    for i, key in enumerate(sigt):
        if burdent[i] > 0 and key in singleton and singleton[key] in mapping:
            candidate[i] = mapping[singleton[key]]
            matched += 1
    metrics = {"accepted_classes": len(mapping), "pair_groups": len(pairs),
               "structural_rows": total, "structural_accuracy": correct / total,
               "min_mapping_purity": min(purities), "min_mapping_support": min(supports),
               "test_matches": matched, "test_changes": int(np.sum(candidate != base))}
    return candidate, metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opt, opt_test, classes, y, _, _, _, _ = v92.load()
    _, v97t = v101.recover_v97(classes)
    train = pd.read_csv(DATA / "train.csv", low_memory=False).set_index("ID").reindex(opt.ID)
    test = pd.read_csv(DATA / "test.csv", low_memory=False).set_index("ID").reindex(opt_test.ID)
    genes = test.columns.tolist()
    candidates, metrics = {}, {}
    for name, variant in [("variant", True), ("gene_set", False)]:
        sig, burden = v119.signature(train, genes, variant)
        sigt, burdent = v119.signature(test, genes, variant)
        candidates[name], metrics[name] = infer(sig, burden, sigt, burdent, y, v97t)
    agree = ((candidates["variant"] == candidates["gene_set"])
             & (candidates["variant"] != v97t))
    disagree = candidates["variant"] != candidates["gene_set"]
    final = v97t.copy()
    final[agree] = candidates["variant"][agree]
    passed = bool(all(m["accepted_classes"] == 4 and m["pair_groups"] >= 400
                      and m["structural_accuracy"] == 1.0
                      and m["min_mapping_purity"] == 1.0 for m in metrics.values())
                  and not disagree.any() and 1 <= agree.sum() <= 10)
    decision = {"candidate_id": "v122_multires_complement_final",
                "parent": "v97_sealed_gene_router_ensemble",
                "parent_oof_macro_f1": 0.6034121303132921,
                "validation_type": "train-only structural leave-one-profile-side-out",
                "metrics": metrics, "resolution_disagreements": int(disagree.sum()),
                "test_changes_vs_v97": int(agree.sum()),
                "changed_ids": opt_test.ID[agree].astype(str).tolist(),
                "promotion_passed": passed, "submission_created": False}
    if passed:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        if not sample.ID.astype(str).equals(opt_test.ID.astype(str)):
            raise ValueError("sample ID order differs")
        sample["SUBCLASS"] = classes[final]
        path = SUBMISSION / ("submission_v122_multires_complement_validated_"
                             + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
        sample.to_csv(path, index=False)
        digest = sha(path)
        duplicates = [str(p) for p in SUBMISSION.glob("*.csv") if p != path and sha(p) == digest]
        if duplicates:
            path.unlink()
            raise ValueError(f"duplicate submission: {duplicates}")
        decision.update({"submission_created": True, "submission_path": str(path),
                         "submission_sha256": digest})
    pd.DataFrame({"ID": opt_test.ID, "v97_label": classes[v97t],
                  "candidate_label": classes[final], "changed": agree}).to_csv(
                      OUT / "v122_test_labels.csv", index=False)
    (OUT / "v122_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
