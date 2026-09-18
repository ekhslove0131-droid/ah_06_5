"""Audited team-derived row semantics without training or test access."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


_SIMPLE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")


def literal_gene_binary(frame: pd.DataFrame, gene_cols: list[str]) -> np.ndarray:
    if frame.columns.intersection(gene_cols).tolist() != gene_cols:
        raise ValueError("gene columns are missing or reordered")
    return (frame[gene_cols].notna() & frame[gene_cols].ne("WT")).to_numpy(dtype=np.float32)


def exact_cell_token_rows(frame: pd.DataFrame, gene_cols: list[str]) -> list[list[str]]:
    binary = literal_gene_binary(frame, gene_cols).astype(bool)
    values = frame[gene_cols].to_numpy(dtype=object, copy=False)
    rows = []
    for row_index in range(len(frame)):
        rows.append([
            f"{gene_cols[column_index]}={values[row_index, column_index]}"
            for column_index in np.flatnonzero(binary[row_index])
        ])
    return rows


def recurrent_tokens(frame: pd.DataFrame, gene_cols: list[str], min_count: int = 3) -> dict[str, set[str]]:
    if not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    result = {}
    for gene in gene_cols:
        values = frame[gene].dropna()
        values = values[values.ne("WT")].astype(str)
        counts = values.value_counts()
        result[gene] = set(counts[counts >= min_count].index.tolist())
    return result


def mutation_severity(value, hot: set[str] | None = None) -> float:
    if pd.isna(value):
        return -1.0
    text = str(value).strip()
    if text == "WT":
        return 0.0
    parts = text.split()
    score = 0.0
    for part in parts:
        if "fs" in part or "*" in part:
            value_score = 3.0
        else:
            match = _SIMPLE.match(part)
            value_score = 1.0 if match and match.group(1) == match.group(3) else 2.0
        score = max(score, value_score)
    if hot and text in hot:
        score = 4.0
    return min(4.0, score + 0.15 * max(0, len(parts) - 1))

