"""Exhaustive two-anchor Round 0 ensemble grid."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
import hashlib
import json
import math

import numpy as np
from sklearn.metrics import f1_score

from slot_selector import SLOT_ORDER, select_slot_candidates


@dataclass(frozen=True)
class MetaCandidate:
    recipe_id: str
    round_index: int
    anchor_id: str
    slot_candidate_ids: tuple[str, ...]
    weights: tuple[float, ...]
    pooling: str
    probability: np.ndarray
    macro_f1: float
    probability_sha256: str
    bias_strength: float = 0.0
    bias: tuple[float, ...] = ()


def enumerate_simplex(*, step, slots):
    if step != 0.25 or not isinstance(slots, int) or slots < 1:
        raise ValueError("V1 simplex requires step 0.25 and positive slots")
    units = 4
    output = []

    def visit(prefix, remaining):
        if len(prefix) == slots - 1:
            output.append(tuple(value / units for value in prefix + [remaining]))
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value)

    visit([], units)
    expected = math.comb(slots + units - 1, units)
    if len(output) != expected:
        raise AssertionError("simplex enumeration count mismatch")
    return output


def _normalize_probability(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not len(value) or value.shape[1] < 2:
        raise ValueError("probability shape is invalid")
    if not np.isfinite(value).all() or (value < 0).any() or np.any(value.sum(axis=1) <= 0):
        raise ValueError("probability values are invalid")
    return value / value.sum(axis=1, keepdims=True)


def pool_probabilities(probabilities, weights, pooling):
    weights = tuple(float(value) for value in weights)
    if len(probabilities) != len(weights) or any(value < 0 for value in weights) or abs(sum(weights) - 1) > 1e-12:
        raise ValueError("pool weights are invalid")
    active = [(probability, weight) for probability, weight in zip(probabilities, weights) if weight > 0]
    normalized = [(_normalize_probability(probability), weight) for probability, weight in active]
    shapes = {probability.shape for probability, _ in normalized}
    if len(shapes) != 1:
        raise ValueError("pool probability shapes differ")
    if pooling == "arithmetic":
        output = sum(weight * probability for probability, weight in normalized)
    elif pooling == "geometric":
        logits = sum(weight * np.log(np.maximum(probability, 1e-12)) for probability, weight in normalized)
        logits -= logits.max(axis=1, keepdims=True)
        output = np.exp(logits)
    else:
        raise ValueError("unknown pooling rule")
    return output / output.sum(axis=1, keepdims=True)


def _sha_probability(probability):
    return hashlib.sha256(np.ascontiguousarray(probability, dtype="<f8").tobytes()).hexdigest()


def _recipe_id(payload):
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return "R0-" + hashlib.sha256(encoded).hexdigest()


def _compare_round0(left, right):
    if abs(left.macro_f1 - right.macro_f1) > 1e-12:
        return -1 if left.macro_f1 > right.macro_f1 else 1
    left_nonzero = sum(value > 0 for value in left.weights)
    right_nonzero = sum(value > 0 for value in right.weights)
    if left_nonzero != right_nonzero:
        return -1 if left_nonzero < right_nonzero else 1
    left_pool = 0 if left.pooling == "arithmetic" else 1
    right_pool = 0 if right.pooling == "arithmetic" else 1
    left_key = (left_pool, left.weights, left.slot_candidate_ids, left.anchor_id, left.recipe_id)
    right_key = (right_pool, right.weights, right.slot_candidate_ids, right.anchor_id, right.recipe_id)
    return -1 if left_key < right_key else (1 if left_key > right_key else 0)


def rank_round0(records):
    return sorted(records, key=cmp_to_key(_compare_round0))


def run_round0(anchors, bank_by_slot, y, guard_mask):
    if len(anchors) != 2:
        raise ValueError("Round 0 requires exactly two anchor branches")
    y = np.asarray(y)
    guard_mask = np.asarray(guard_mask)
    weights_grid = enumerate_simplex(step=0.25, slots=5)
    records = []
    for anchor_id in sorted(anchors):
        anchor = _normalize_probability(anchors[anchor_id])
        if y.shape != (len(anchor),) or guard_mask.shape != (len(anchor),) or guard_mask.dtype != np.bool_:
            raise ValueError("Round 0 identity arrays are misaligned")
        selected = select_slot_candidates(anchor, bank_by_slot, y, guard_mask)
        candidate_ids = tuple(selected[slot]["candidate_id"] for slot in SLOT_ORDER)
        probabilities = [anchor] + [selected[slot]["probability"] for slot in SLOT_ORDER]
        for weights in weights_grid:
            for pooling in ("arithmetic", "geometric"):
                probability = pool_probabilities(probabilities, weights, pooling)
                probability[guard_mask] = anchor[guard_mask]
                probability /= probability.sum(axis=1, keepdims=True)
                score = float(f1_score(
                    y, probability.argmax(axis=1), labels=np.arange(probability.shape[1]),
                    average="macro", zero_division=0,
                ))
                payload = {
                    "schema_version": "ROUND0_RECIPE_V1",
                    "anchor_id": anchor_id,
                    "slot_candidate_ids": candidate_ids,
                    "weights": weights,
                    "pooling": pooling,
                }
                records.append(MetaCandidate(
                    recipe_id=_recipe_id(payload), round_index=0, anchor_id=anchor_id,
                    slot_candidate_ids=candidate_ids, weights=weights, pooling=pooling,
                    probability=probability, macro_f1=score,
                    probability_sha256=_sha_probability(probability),
                ))
    if len(records) != 280 or len({record.recipe_id for record in records}) != 280:
        raise AssertionError("Round 0 must contain exactly 280 unique recipes")
    return records
