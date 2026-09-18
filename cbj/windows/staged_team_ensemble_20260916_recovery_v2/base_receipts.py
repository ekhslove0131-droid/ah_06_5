"""Canonical hashes and completion checks for base-bank jobs."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def canonical_sha256(value):
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sequence_sha256(value):
    return canonical_sha256(np.asarray(value).tolist())


def probability_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def validate_oof(probability, coverage, rows, classes):
    probability = np.asarray(probability)
    coverage = np.asarray(coverage)
    if probability.shape != (rows, classes):
        raise ValueError("base OOF shape mismatch")
    if coverage.shape != (rows,) or not np.all(coverage == 1):
        raise ValueError("base OOF coverage is not exactly once")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("base OOF contains invalid probability")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("base OOF rows do not normalize")

