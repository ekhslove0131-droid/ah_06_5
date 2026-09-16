"""Label-free raw annotation count view for the CBJ H1 development experiment."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mcmp_repr as M


TYPE_KINDS = (
    "missense", "synonymous", "frameshift", "stop_gained", "stop_lost",
    "deletion", "insertion", "delins", "complex_unparsed",
)

RAW_NAMES = (
    "raw_annotation|total_token_count",
    *(f"raw_annotation|type|{kind}_count" for kind in TYPE_KINDS),
    "raw_annotation|multi_annotation_gene_count",
    "raw_annotation|within_gene_excess_count",
)


def build_raw_annotation_view(frame, gene_columns):
    if list(frame.columns) != list(gene_columns):
        raise ValueError("Raw annotation frame/gene order mismatch")
    values = np.zeros((len(frame), len(RAW_NAMES)), dtype=np.float32)
    kind_index = {kind: 1 + index for index, kind in enumerate(TYPE_KINDS)}
    total_tokens = 0
    patients_with_multi = 0
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        multi_genes = 0
        excess = 0
        for raw in row:
            tokens = M.split_tokens(raw)
            values[row_index, 0] += len(tokens)
            total_tokens += len(tokens)
            if len(tokens) >= 2:
                multi_genes += 1
                excess += len(tokens) - 1
            for token in tokens:
                kind = M.classify_token(token)[0]
                if kind not in kind_index:
                    raise ValueError("Parser returned an unknown annotation type")
                values[row_index, kind_index[kind]] += 1
        values[row_index, -2] = multi_genes
        values[row_index, -1] = excess
        patients_with_multi += int(multi_genes > 0)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Raw annotation view produced invalid counts")
    return sparse.csr_matrix(values), list(RAW_NAMES), {
        "rows": len(frame), "genes": len(gene_columns), "total_tokens": total_tokens,
        "patients_with_multi_annotation_gene": patients_with_multi,
        "label_access": False, "parser": "mcmp_repr.split_tokens/classify_token",
    }


def append_raw_view(cache: dict, raw_view, raw_names):
    raw_view = raw_view.tocsr()
    train_indices = np.asarray(cache["train_indices"])
    valid_indices = np.asarray(cache["valid_indices"])
    if raw_view.shape[1] != len(raw_names):
        raise ValueError("Raw view/name mismatch")
    if len(set(raw_names)) != len(raw_names) or set(raw_names) & set(cache["names"]):
        raise ValueError("Raw feature names duplicate existing features")
    if max(train_indices.max(), valid_indices.max()) >= raw_view.shape[0]:
        raise ValueError("Raw view row identity out of range")
    result = dict(cache)
    result["train"] = sparse.hstack(
        [cache["train"], raw_view[train_indices]], format="csr", dtype=np.float32)
    result["valid"] = sparse.hstack(
        [cache["valid"], raw_view[valid_indices]], format="csr", dtype=np.float32)
    result["names"] = list(cache["names"]) + list(raw_names)
    return result


def validate_h1_config(config: dict) -> None:
    if config.get("version") != "cbj-goal055-h1-raw12-v1":
        raise ValueError("Wrong H1 experiment version")
    if config.get("seed") != 43 or config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("H1 is restricted to five seed43 folds")
    if config.get("fit_budget") != 5 or config.get("weight_alpha") != 0.5:
        raise ValueError("H1 requires five fits and fixed alpha 0.5")
    if config.get("raw_feature_names") != list(RAW_NAMES):
        raise ValueError("H1 raw feature contract must contain the exact 12 names")
    expected_stats = config.get("expected_raw_stats")
    if not isinstance(expected_stats, dict) or set(expected_stats) != {
            "total_tokens", "patients_with_multi_annotation_gene"}:
        raise ValueError("H1 expected raw aggregate contract is missing")
    if config.get("model") != config.get("baseline_model"):
        raise ValueError("Raw12 append must be the sole variable from baseline_model")
    if {"early_stopping_rounds", "callbacks"} & set(config.get("model", {})):
        raise ValueError("Early stopping and callbacks are forbidden")
    if any(config.get(key) is not False for key in ("test_read", "server_score_used", "submission")):
        raise ValueError("Forbidden data flow or external action enabled")


def validate_raw_stats(observed: dict, expected: dict) -> None:
    for key in ("total_tokens", "patients_with_multi_annotation_gene"):
        if observed.get(key) != expected.get(key):
            raise ValueError(f"Raw annotation aggregate mismatch: {key}")
