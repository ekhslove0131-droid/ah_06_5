"""Pure, fail-closed primitives for the fixed seven-model stacking bundle."""

from __future__ import annotations

import copy
import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


_REPRESENTATIONS = (
    "probability",
    "log_probability",
    "log_probability_confidence",
)
_C_VALUES = (0.001, 0.01, 0.1, 1, 10, 100)
_WEIGHTINGS = ("none", "sqrt_inverse", "balanced")
_POOLINGS = ("arithmetic", "geometric")
_MODEL_COUNT = 7
_LOG_FLOOR = 1e-12
_THREAD_LIMIT = 4
_NORMALIZATION_ATOL = 1e-8
_VALIDATION_EVIDENCE = "CACHED_BASE_OOF_META_CV_ADAPTIVE_NOT_NESTED"


def declarations() -> list[dict]:
    """Return the complete, deterministically ordered 54-item search grid."""

    return [
        {"representation": representation, "C": c_value, "weighting": weighting}
        for representation in _REPRESENTATIONS
        for c_value in _C_VALUES
        for weighting in _WEIGHTINGS
    ]


def _as_probability_tensor(p) -> np.ndarray:
    try:
        probability = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("p must be a numeric probability tensor") from exc
    if probability.ndim != 3:
        raise ValueError("p must have shape (7, N, K)")
    if probability.shape[0] != _MODEL_COUNT:
        raise ValueError("p must contain exactly seven models")
    if probability.shape[1] < 1 or probability.shape[2] < 2:
        raise ValueError("p must contain at least one row and two classes")
    if not np.isfinite(probability).all():
        raise ValueError("p must contain only finite probabilities")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("p probabilities must lie in [0, 1]")
    row_sums = probability.sum(axis=2)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=_NORMALIZATION_ATOL):
        raise ValueError("every p probability row must be normalized")
    return probability


def _as_probability_matrix(value, name: str) -> np.ndarray:
    try:
        probability = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric probability matrix") from exc
    if probability.ndim != 2 or probability.shape[0] < 1 or probability.shape[1] < 2:
        raise ValueError(f"{name} must have shape (N, K) with N >= 1 and K >= 2")
    if not np.isfinite(probability).all():
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    if not np.allclose(
        probability.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=_NORMALIZATION_ATOL,
    ):
        raise ValueError(f"every {name} probability row must be normalized")
    return probability


def features(p, representation: str):
    """Build row-only float64 meta-features from a (7, N, K) tensor."""

    if representation not in _REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation!r}")
    probability = _as_probability_tensor(p)
    rows = probability.shape[1]
    probability_features = probability.transpose(1, 0, 2).reshape(rows, -1).copy()
    if representation == "probability":
        return probability_features

    clipped = np.clip(probability, _LOG_FLOOR, 1.0)
    log_features = np.log(clipped).transpose(1, 0, 2).reshape(rows, -1).copy()
    if representation == "log_probability":
        return log_features

    entropy = -np.sum(probability * np.log(clipped), axis=2)
    maximum = np.max(probability, axis=2)
    top_two = np.partition(probability, kth=-2, axis=2)[:, :, -2:]
    margin = top_two[:, :, 1] - top_two[:, :, 0]
    confidence = np.stack((entropy, maximum, margin), axis=2)
    confidence_features = confidence.transpose(1, 0, 2).reshape(rows, -1).copy()
    return np.concatenate((log_features, confidence_features), axis=1)


def _validated_classes(classes: int) -> int:
    if isinstance(classes, (bool, np.bool_)) or not isinstance(classes, Integral):
        raise TypeError("classes must be an integer")
    value = int(classes)
    if value < 2:
        raise ValueError("classes must be at least two")
    return value


def _validated_labels(y, classes: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size < 1:
        raise ValueError("y must be a non-empty one-dimensional array")
    if labels.dtype.kind not in "iu":
        raise TypeError("y must contain integer labels")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("y labels must be in the integer range [0, classes)")
    counts = np.bincount(labels, minlength=classes)
    if counts.shape[0] != classes or np.any(counts == 0):
        raise ValueError("y must have support for every class")
    return labels, counts.astype(np.float64)


def sample_weights(y, weighting: str, classes: int):
    """Compute train-label-only per-row weights and normalize them to mean one."""

    class_count = _validated_classes(classes)
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"unknown weighting: {weighting!r}")
    labels, counts = _validated_labels(y, class_count)
    if weighting == "none":
        return np.ones(labels.size, dtype=np.float64)
    if weighting == "balanced":
        class_weights = labels.size / (class_count * counts)
    else:
        class_weights = 1.0 / np.sqrt(counts)
    weights = np.asarray(class_weights[labels], dtype=np.float64)
    weights /= weights.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("computed sample weights must be finite and positive")
    return weights


def _validated_declaration(declaration: dict) -> dict:
    if not isinstance(declaration, dict):
        raise TypeError("declaration must be a dictionary")
    required = {"representation", "C", "weighting"}
    if set(declaration) != required:
        raise ValueError("declaration must contain exactly representation, C, and weighting")
    representation = declaration["representation"]
    weighting = declaration["weighting"]
    c_value = declaration["C"]
    if representation not in _REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation!r}")
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"unknown weighting: {weighting!r}")
    if isinstance(c_value, (bool, np.bool_)) or not isinstance(c_value, Real):
        raise TypeError("C must be a finite positive real number")
    if not np.isfinite(c_value) or float(c_value) <= 0.0:
        raise ValueError("C must be finite and positive")
    return copy.deepcopy(declaration)


def fit_meta(declaration: dict, p, y) -> dict:
    """Fit one declared scaler and L2/lbfgs logistic meta learner."""

    validated_declaration = _validated_declaration(declaration)
    probability = _as_probability_tensor(p)
    class_count = probability.shape[2]
    labels, _ = _validated_labels(y, class_count)
    if labels.size != probability.shape[1]:
        raise ValueError("y length must match the number of probability rows")
    matrix = features(probability, validated_declaration["representation"])
    weights = sample_weights(labels, validated_declaration["weighting"], class_count)
    scaler = StandardScaler(with_mean=True, with_std=True)
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=float(validated_declaration["C"]),
        max_iter=2000,
        tol=1e-5,
        random_state=42,
        n_jobs=1,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with threadpool_limits(limits=_THREAD_LIMIT):
            scaled = scaler.fit_transform(matrix)
            model.fit(scaled, labels, sample_weight=weights)
    convergence_warnings = [
        str(item.message)
        for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    expected_classes = np.arange(class_count, dtype=np.int64)
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError("fitted model classes do not match the required integer class order")

    effective_parameters = {
        "execution": "CPU_INTENTIONAL",
        "threadpool_limits": _THREAD_LIMIT,
        "estimator": {
            "penalty": "l2",
            "solver": "lbfgs",
            "C": float(validated_declaration["C"]),
            "max_iter": 2000,
            "tol": 1e-5,
            "random_state": 42,
            "n_jobs": 1,
            "n_iter": model.n_iter_.astype(int).tolist(),
        },
        "scaler": {"with_mean": True, "with_std": True},
        "weighting": validated_declaration["weighting"],
        "validation_evidence": _VALIDATION_EVIDENCE,
    }
    return {
        "scaler": scaler,
        "model": model,
        "declaration": validated_declaration,
        "classes": expected_classes.tolist(),
        "feature_count": int(matrix.shape[1]),
        "effective_parameters": effective_parameters,
        "convergence_warnings": convergence_warnings,
    }


def _artifact_class_order(artifact: dict, input_classes: int) -> np.ndarray:
    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dictionary")
    required = {
        "scaler",
        "model",
        "declaration",
        "classes",
        "feature_count",
        "effective_parameters",
        "convergence_warnings",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"artifact is missing required fields: {sorted(missing)}")
    artifact_classes = np.asarray(artifact["classes"])
    expected = np.arange(input_classes, dtype=np.int64)
    if artifact_classes.ndim != 1 or artifact_classes.dtype.kind not in "iu":
        raise ValueError("artifact classes must be a one-dimensional integer order")
    if not np.array_equal(artifact_classes.astype(np.int64, copy=False), expected):
        raise ValueError("artifact classes must be the exact integer order 0..K-1")
    return expected


def predict_meta(artifact: dict, p):
    """Predict normalized probabilities aligned to integer class order 0..K-1."""

    probability = _as_probability_tensor(p)
    expected_classes = _artifact_class_order(artifact, probability.shape[2])
    declaration = _validated_declaration(artifact["declaration"])
    matrix = features(probability, declaration["representation"])
    feature_count = artifact["feature_count"]
    if isinstance(feature_count, (bool, np.bool_)) or not isinstance(feature_count, Integral):
        raise TypeError("artifact feature_count must be an integer")
    if int(feature_count) != matrix.shape[1]:
        raise ValueError("artifact feature_count does not match input features")

    model_classes = np.asarray(getattr(artifact["model"], "classes_", None))
    if model_classes.ndim != 1 or model_classes.dtype.kind not in "iu":
        raise ValueError("model classes must be a one-dimensional integer array")
    model_classes = model_classes.astype(np.int64, copy=False)
    if model_classes.size != expected_classes.size or not np.array_equal(
        np.sort(model_classes), expected_classes
    ):
        raise ValueError("model classes are missing or do not match artifact classes")

    with threadpool_limits(limits=_THREAD_LIMIT):
        scaled = artifact["scaler"].transform(matrix)
        raw = np.asarray(artifact["model"].predict_proba(scaled), dtype=np.float64)
    if raw.shape != (probability.shape[1], model_classes.size):
        raise ValueError("model probability output shape does not match model classes")
    if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("model returned invalid probabilities")
    raw_row_sums = raw.sum(axis=1)
    if not np.allclose(raw_row_sums, 1.0, rtol=0.0, atol=_NORMALIZATION_ATOL):
        raise ValueError("model probabilities are not normalized")
    aligned = np.empty((raw.shape[0], expected_classes.size), dtype=np.float64)
    for source_column, class_index in enumerate(model_classes):
        aligned[:, class_index] = raw[:, source_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0.0):
        raise ValueError("model returned non-normalizable probabilities")
    aligned /= row_sums
    if not np.allclose(aligned.sum(axis=1), 1.0, rtol=0.0, atol=1e-15):
        raise ValueError("normalized model probabilities do not sum to one")
    return aligned


def blend(baseline, meta, alpha: float, pooling: str, guard):
    """Blend two probability matrices, then restore guarded rows to baseline."""

    baseline_probability = _as_probability_matrix(baseline, "baseline")
    meta_probability = _as_probability_matrix(meta, "meta")
    if baseline_probability.shape != meta_probability.shape:
        raise ValueError("baseline and meta probability shapes must match")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite real number")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if pooling not in _POOLINGS:
        raise ValueError(f"unknown pooling: {pooling!r}")
    guard_mask = np.asarray(guard)
    if guard_mask.dtype.kind != "b" or guard_mask.shape != (baseline_probability.shape[0],):
        raise ValueError("guard must be a boolean vector with one value per row")

    if alpha_value == 0.0:
        output = baseline_probability.copy()
    elif alpha_value == 1.0:
        output = meta_probability.copy()
    elif pooling == "arithmetic":
        output = (1.0 - alpha_value) * baseline_probability + alpha_value * meta_probability
        output /= output.sum(axis=1, keepdims=True)
    else:
        baseline_log = np.log(np.clip(baseline_probability, _LOG_FLOOR, 1.0))
        meta_log = np.log(np.clip(meta_probability, _LOG_FLOOR, 1.0))
        output = np.exp((1.0 - alpha_value) * baseline_log + alpha_value * meta_log)
        output /= output.sum(axis=1, keepdims=True)

    output[guard_mask] = baseline_probability[guard_mask]
    if not np.isfinite(output).all() or np.any(output < 0.0) or np.any(output > 1.0):
        raise ValueError("blend produced invalid probabilities")
    if not np.allclose(output.sum(axis=1), 1.0, rtol=0.0, atol=_NORMALIZATION_ATOL):
        raise ValueError("blend produced non-normalized probabilities")
    return np.asarray(output, dtype=np.float64)
