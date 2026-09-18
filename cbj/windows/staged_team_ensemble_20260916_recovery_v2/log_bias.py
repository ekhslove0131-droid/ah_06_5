"""Bounded deterministic macro-F1 class log-bias optimizer."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _validate(probability, y):
    probability = np.asarray(probability, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if probability.ndim != 2 or y.shape != (len(probability),) or probability.shape[1] < 2:
        raise ValueError("log-bias probability/label shape mismatch")
    if not np.isfinite(probability).all() or (probability < 0).any() or np.any(probability.sum(axis=1) <= 0):
        raise ValueError("log-bias probabilities are invalid")
    if set(np.unique(y).tolist()) != set(range(probability.shape[1])):
        raise ValueError("log-bias labels must contain every class")
    return probability / probability.sum(axis=1, keepdims=True), y


def apply_log_bias(probability, bias, epsilon=1e-300):
    probability = np.asarray(probability, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if probability.ndim != 2 or bias.shape != (probability.shape[1],):
        raise ValueError("log-bias application shape mismatch")
    logits = np.log(np.maximum(probability, epsilon)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=1, keepdims=True)


def _state(probability, y, bias, penalty=0.001):
    prediction = apply_log_bias(probability, bias).argmax(axis=1)
    macro = float(f1_score(y, prediction, labels=np.arange(probability.shape[1]), average="macro", zero_division=0))
    norm = float(np.mean(np.square(bias)))
    return {"objective": macro - penalty * norm, "macro_f1": macro, "mean_bias_squared": norm}


def _better(candidate, incumbent, tolerance=1e-12):
    for name, higher in (
        ("objective", True), ("macro_f1", True), ("mean_bias_squared", False),
        ("abs_delta", False), ("delta", False),
    ):
        if abs(candidate[name] - incumbent[name]) <= tolerance:
            continue
        return candidate[name] > incumbent[name] if higher else candidate[name] < incumbent[name]
    return False


def fit_c10_log_bias(probability, y):
    probability, y = _validate(probability, y)
    classes = probability.shape[1]
    bias = np.zeros(classes, dtype=np.float64)
    current = _state(probability, y, bias)
    initial = dict(current)
    history = []
    passes = 0
    for pass_index in range(4):
        passes += 1
        commits = 0
        for coordinate in range(classes):
            incumbent = {**current, "delta": 0.0, "abs_delta": 0.0, "bias": bias.copy()}
            best = incumbent
            for integer_delta in range(-5, 6):
                delta = integer_delta / 10
                candidate_bias = bias.copy(); candidate_bias[coordinate] += delta
                candidate_bias -= candidate_bias.mean()
                if np.max(np.abs(candidate_bias)) > 1.5 + 1e-12:
                    continue
                candidate = {**_state(probability, y, candidate_bias), "delta": delta, "abs_delta": abs(delta), "bias": candidate_bias}
                if _better(candidate, best):
                    best = candidate
            if best["objective"] > current["objective"] + 1e-12:
                previous = current["objective"]
                bias = best["bias"]
                current = {name: best[name] for name in ("objective", "macro_f1", "mean_bias_squared")}
                history.append({
                    "pass": pass_index, "coordinate": coordinate, "delta": best["delta"],
                    "objective_before": previous, "objective_after": current["objective"],
                })
                commits += 1
        if commits == 0:
            break
    final = _state(probability, y, bias)
    if abs(float(bias.mean())) > 1e-12 or np.max(np.abs(bias)) > 1.5 + 1e-12:
        raise ValueError("log-bias optimizer invariant failed")
    return bias, {
        "schema_version": "BOUNDED_LOG_BIAS_RECEIPT_V1",
        "initial": initial, "final": final, "centered_bias": bias.tolist(),
        "history": history, "commits": len(history), "passes_attempted": passes,
        "outer_validation_labels_accepted": False,
    }

