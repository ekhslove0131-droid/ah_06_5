"""Fold-local label-free global29 plus raw annotation12 numeric view."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from feature_transforms import _mcmp_block_pair, _mcmp_rows
from vendor import mcmp_repr


TYPE_KINDS = (
    "missense", "synonymous", "frameshift", "stop_gained", "stop_lost",
    "deletion", "insertion", "delins", "complex_unparsed",
)
FULL_BLOCKS = ["gene", "type", "gene_change", "aa", "aa_from", "aa_to", "global", "site", "exact"]
RAW_NAMES = (
    "raw_annotation|total_token_count",
    *(f"raw_annotation|type|{kind}_count" for kind in TYPE_KINDS),
    "raw_annotation|multi_annotation_gene_count",
    "raw_annotation|within_gene_excess_count",
)


def _column(matrix, index, name):
    if name not in index:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.asarray(matrix[:, index[name]].toarray()).ravel().astype(np.float32)


def _sum_columns(matrix, indices):
    if not indices:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.asarray(matrix[:, indices].sum(axis=1)).ravel().astype(np.float32)


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).astype(np.float32)


def _global29(matrix, names, gene_count):
    index = {name: position for position, name in enumerate(names)}
    mutation = _column(matrix, index, "global|mutation_gene_count")
    missing = _column(matrix, index, "global|missing_gene_count")
    observed = _column(matrix, index, "global|observed_gene_count")
    type_counts = {
        kind: _sum_columns(matrix, [
            position for position, name in enumerate(names)
            if name.startswith("type|") and name.endswith("|" + kind)
        ])
        for kind in TYPE_KINDS
    }
    total = sum(type_counts.values(), np.zeros(matrix.shape[0], dtype=np.float32))
    truncating = type_counts["frameshift"] + type_counts["stop_gained"]
    indel = type_counts["deletion"] + type_counts["insertion"] + type_counts["delins"]
    parsed = total - type_counts["complex_unparsed"]
    columns = [
        ("summary|log1p_mutated_gene_count", np.log1p(mutation).astype(np.float32)),
        ("summary|mutation_fraction_observed", _ratio(mutation, observed)),
        ("summary|missing_fraction_panel", missing / float(gene_count)),
        ("summary|type|total_token_count", total),
        ("summary|type|multi_annotation_excess", np.maximum(total - mutation, 0)),
        ("summary|type|multi_annotation_fraction", _ratio(np.maximum(total - mutation, 0), total)),
        ("summary|type|parsed_fraction", _ratio(parsed, total)),
        ("summary|type|truncating_count", truncating),
        ("summary|type|truncating_fraction", _ratio(truncating, total)),
        ("summary|type|indel_count", indel),
        ("summary|type|indel_fraction", _ratio(indel, total)),
    ]
    for kind in TYPE_KINDS:
        columns.append((f"summary|type|{kind}_count", type_counts[kind]))
        columns.append((f"summary|type|{kind}_fraction", _ratio(type_counts[kind], total)))
    values = np.column_stack([value for _, value in columns]).astype(np.float32)
    return values, [name for name, _ in columns]


def _raw12(frame, gene_cols):
    values = np.zeros((len(frame), len(RAW_NAMES)), dtype=np.float32)
    kind_index = {kind: 1 + index for index, kind in enumerate(TYPE_KINDS)}
    for row_index, row in enumerate(frame[gene_cols].itertuples(index=False, name=None)):
        multi_genes = 0
        excess = 0
        for raw in row:
            tokens = mcmp_repr.split_tokens(raw)
            values[row_index, 0] += len(tokens)
            if len(tokens) >= 2:
                multi_genes += 1
                excess += len(tokens) - 1
            for token in tokens:
                values[row_index, kind_index[mcmp_repr.classify_token(token)[0]]] += 1
        values[row_index, -2] = multi_genes
        values[row_index, -1] = excess
    return values


def build_numeric41(train, valid, gene_cols, train_groups):
    if list(train.columns) != list(gene_cols) or list(valid.columns) != list(gene_cols):
        raise ValueError("numeric41 gene columns are missing or reordered")
    train_rows, valid_rows = _mcmp_rows(train, valid, gene_cols)
    encoded_train, encoded_valid, encoder = _mcmp_block_pair(
        train_rows, valid_rows, train_groups, FULL_BLOCKS, 3,
    )
    global_train, global_names = _global29(encoded_train, encoder["names"], len(gene_cols))
    global_valid, valid_names = _global29(encoded_valid, encoder["names"], len(gene_cols))
    if global_names != valid_names or len(global_names) != 29:
        raise ValueError("numeric global29 contract changed")
    numeric_train = np.column_stack([global_train, _raw12(train, gene_cols)]).astype(np.float32)
    numeric_valid = np.column_stack([global_valid, _raw12(valid, gene_cols)]).astype(np.float32)
    if numeric_train.shape[1] != 41 or not np.isfinite(numeric_train).all() or (numeric_train < 0).any():
        raise ValueError("numeric41 train values are invalid")
    if not np.isfinite(numeric_valid).all() or (numeric_valid < 0).any():
        raise ValueError("numeric41 valid values are invalid")
    return {
        "train": numeric_train,
        "valid": numeric_valid,
        "names": global_names + list(RAW_NAMES),
        "fitted": {"encoder": encoder, "global_names": global_names, "raw_names": list(RAW_NAMES)},
        "receipt": {
            "schema_version": "NUMERIC41_RECEIPT_V1",
            "train_rows": len(train), "valid_rows": len(valid), "width": 41,
            "fit_scope": "partition training rows/groups only",
            "labels_accepted": False, "test_read": False,
        },
    }

