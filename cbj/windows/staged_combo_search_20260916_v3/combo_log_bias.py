"""Exact-compatible, allocation-light macro-F1 class log-bias optimizer.

The public optimizer preserves the bounded four-pass coordinate search and V1
receipt emitted by the recovery-v2 oracle.  Its hot path caches normalized
log-probabilities and computes macro-F1 directly from integer confusion counts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


_TIE_FALLBACK_GAP = 1e-12


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


def _exact_argmax_from_cached_logits(base_logits, bias):
    """Match apply_log_bias(...).argmax(), including floating-point ties.

    Argmax on logits is equivalent for ordinary rows.  Rows whose two leading
    logits are close enough for subtract/exp/divide rounding to create a tie
    are replayed through the oracle's exact softmax operation sequence.
    """

    logits = base_logits + bias
    prediction = logits.argmax(axis=1)
    maximum = logits[np.arange(len(logits)), prediction]
    near_tie = np.sum(maximum[:, None] - logits <= _TIE_FALLBACK_GAP, axis=1) > 1
    if np.any(near_tie):
        replay = logits[near_tie].copy()
        replay -= replay.max(axis=1, keepdims=True)
        replay = np.exp(replay)
        replay /= replay.sum(axis=1, keepdims=True)
        prediction[near_tie] = replay.argmax(axis=1)
    return prediction


def _macro_f1_from_predictions(y, prediction, classes):
    confusion = np.bincount(y * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(classes, dtype=np.float64),
        where=denominator != 0,
    )
    return float(per_class.mean())


def _state_from_cached_logits(base_logits, y, bias, penalty=0.001):
    prediction = _exact_argmax_from_cached_logits(base_logits, bias)
    macro = _macro_f1_from_predictions(y, prediction, base_logits.shape[1])
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
    base_logits = np.log(np.maximum(probability, 1e-300))
    bias = np.zeros(classes, dtype=np.float64)
    current = _state_from_cached_logits(base_logits, y, bias)
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
                candidate_bias = bias.copy()
                candidate_bias[coordinate] += delta
                candidate_bias -= candidate_bias.mean()
                if np.max(np.abs(candidate_bias)) > 1.5 + 1e-12:
                    continue
                candidate = {
                    **_state_from_cached_logits(base_logits, y, candidate_bias),
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "bias": candidate_bias,
                }
                if _better(candidate, best):
                    best = candidate
            if best["objective"] > current["objective"] + 1e-12:
                previous = current["objective"]
                bias = best["bias"]
                current = {name: best[name] for name in ("objective", "macro_f1", "mean_bias_squared")}
                history.append({
                    "pass": pass_index,
                    "coordinate": coordinate,
                    "delta": best["delta"],
                    "objective_before": previous,
                    "objective_after": current["objective"],
                })
                commits += 1
        if commits == 0:
            break
    final = _state_from_cached_logits(base_logits, y, bias)
    if abs(float(bias.mean())) > 1e-12 or np.max(np.abs(bias)) > 1.5 + 1e-12:
        raise ValueError("log-bias optimizer invariant failed")
    return bias, {
        "schema_version": "BOUNDED_LOG_BIAS_RECEIPT_V1",
        "initial": initial,
        "final": final,
        "centered_bias": bias.tolist(),
        "history": history,
        "commits": len(history),
        "passes_attempted": passes,
        "outer_validation_labels_accepted": False,
    }


def optimizer_receipt():
    source = Path(__file__).read_bytes()
    return {
        "schema_version": "LOG_BIAS_OPTIMIZER_RECEIPT_V1",
        "optimizer_version": "exact-confusion-v1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "oracle_receipt_compatible": True,
    }


def fit_c10_log_bias_with_optimizer_receipt(probability, y):
    bias, compatible_receipt = fit_c10_log_bias(probability, y)
    return bias, compatible_receipt, optimizer_receipt()
