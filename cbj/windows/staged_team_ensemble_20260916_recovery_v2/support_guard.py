"""Stable equal geometric pooling and feature-only anchor fallback."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _numeric_zero_rows(matrix):
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return np.diff(matrix.indptr) == 0


def support_guard_mask(train_symbolic, valid_symbolic):
    """Return the frozen feature-only fallback mask.

    A held-out row is guarded only when its symbolic vector is all-zero and
    no all-zero symbolic row occurred in the fitting partition.  Labels and
    row identities are deliberately absent from this interface.
    """
    if not sparse.issparse(train_symbolic) or not sparse.issparse(valid_symbolic):
        raise ValueError("support guard requires sparse symbolic matrices")
    if train_symbolic.shape[1] != valid_symbolic.shape[1]:
        raise ValueError("support guard train/valid columns differ")
    train_zero = _numeric_zero_rows(train_symbolic)
    valid_zero = _numeric_zero_rows(valid_symbolic)
    mask = valid_zero.copy() if not train_zero.any() else np.zeros(len(valid_zero), dtype=bool)
    receipt = {
        "train_rows": int(train_symbolic.shape[0]),
        "valid_rows": int(valid_symbolic.shape[0]),
        "symbolic_features": int(train_symbolic.shape[1]),
        "train_zero_rows": int(train_zero.sum()),
        "valid_zero_rows": int(valid_zero.sum()),
        "guarded_rows": int(mask.sum()),
        "rule": "guard valid numeric-allzero iff train numeric-allzero count equals zero",
        "labels_or_ids_accepted_by_rule": False,
    }
    return mask, receipt


def _probability(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} probability is invalid")
    total = value.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError(f"{name} probability has zero mass")
    return value / total


def geometric_pool(left, right, epsilon=1e-300):
    left = _probability(left, "left")
    right = _probability(right, "right")
    if left.shape != right.shape:
        raise ValueError("geometric pool shape mismatch")
    logits = 0.5 * np.log(np.maximum(left, epsilon)) + 0.5 * np.log(np.maximum(right, epsilon))
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=1, keepdims=True)


def guarded_probability(candidate, h1_probability, guard_mask):
    candidate = _probability(candidate, "candidate")
    h1_probability = _probability(h1_probability, "H1")
    guard_mask = np.asarray(guard_mask)
    if candidate.shape != h1_probability.shape or guard_mask.shape != (len(candidate),) or guard_mask.dtype != np.bool_:
        raise ValueError("support guard shape or type mismatch")
    output = candidate.copy()
    output[guard_mask] = h1_probability[guard_mask]
    return output
