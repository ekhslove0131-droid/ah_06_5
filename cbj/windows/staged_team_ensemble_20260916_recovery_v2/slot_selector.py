"""Deterministic outer-train-inner-OOF slot selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


SLOT_ORDER = ("project", "kim", "ahn", "cross")


def _macro(y, probability):
    probability = np.asarray(probability, dtype=np.float64)
    return float(f1_score(
        y, probability.argmax(axis=1), labels=np.arange(probability.shape[1]),
        average="macro", zero_division=0,
    ))


def select_slot_candidates(anchor_probability, candidates_by_slot, y, guard_mask):
    anchor_probability = np.asarray(anchor_probability, dtype=np.float64)
    y = np.asarray(y)
    guard_mask = np.asarray(guard_mask)
    if y.shape != (len(anchor_probability),) or guard_mask.shape != (len(y),) or guard_mask.dtype != np.bool_:
        raise ValueError("slot selection inputs are misaligned")
    if set(candidates_by_slot) != set(SLOT_ORDER):
        raise ValueError("slot bank must contain exactly project, kim, ahn and cross")
    result = {}
    for slot in SLOT_ORDER:
        rows = []
        if not candidates_by_slot[slot]:
            raise ValueError(f"slot {slot} is empty")
        for candidate_id, raw in candidates_by_slot[slot].items():
            probability = np.asarray(raw, dtype=np.float64)
            if probability.shape != anchor_probability.shape:
                raise ValueError("candidate probability shape mismatch")
            standalone = _macro(y, probability)
            pooled = (anchor_probability + probability) / 2.0
            pooled[guard_mask] = anchor_probability[guard_mask]
            pooled /= pooled.sum(axis=1, keepdims=True)
            pooled_score = _macro(y, pooled)
            if pooled_score > standalone + 1e-12:
                best_score, mode = pooled_score, "anchor_half_arithmetic_guarded"
            else:
                best_score, mode = standalone, "standalone"
            rows.append((candidate_id, best_score, mode, standalone, pooled_score, probability))
        rows.sort(key=lambda item: (-item[1], item[0]))
        candidate_id, best_score, mode, standalone, pooled_score, probability = rows[0]
        result[slot] = {
            "candidate_id": candidate_id,
            "selection_score": best_score,
            "selection_mode": mode,
            "standalone_macro_f1": standalone,
            "anchor_half_macro_f1": pooled_score,
            "probability": probability,
        }
    return result

