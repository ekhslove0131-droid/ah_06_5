"""Fixed-threshold, greedy-diversity promotion."""

from __future__ import annotations

from functools import cmp_to_key

import numpy as np


def disagreement(left, right):
    left_label = np.asarray(left.probability).argmax(axis=1)
    right_label = np.asarray(right.probability).argmax(axis=1)
    if left_label.shape != right_label.shape:
        raise ValueError("promotion predictions are misaligned")
    return float(np.mean(left_label != right_label))


def _compare(left, right):
    if abs(left.macro_f1 - right.macro_f1) > 1e-12:
        return -1 if left.macro_f1 > right.macro_f1 else 1
    return -1 if left.recipe_id < right.recipe_id else (1 if left.recipe_id > right.recipe_id else 0)


def promote(candidates, *, threshold=0.55, minimum_disagreement=0.03, maximum=8):
    if threshold != 0.55 or minimum_disagreement != 0.03 or maximum != 8:
        raise ValueError("V1 promotion constants are frozen")
    eligible = [candidate for candidate in candidates if candidate.macro_f1 >= threshold]
    eligible.sort(key=cmp_to_key(_compare))
    retained = []
    for candidate in eligible:
        if all(disagreement(candidate, previous) >= minimum_disagreement for previous in retained):
            retained.append(candidate)
            if len(retained) == maximum:
                break
    return retained

