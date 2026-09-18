"""Pure, fail-closed primitives for the fixed seven-model stacking bundle."""

from __future__ import annotations

import copy
import subprocess
import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.preprocessing import StandardScaler


_REPRESENTATIONS = ("probability", "log_probability", "log_probability_confidence")
_SEARCH_REPRESENTATION = "log_probability_confidence"
_DEPTHS = (3, 5, 7)
_ITERATIONS = (200, 600, 1200)
_L2_LEAF_REGS = (3, 30)
_WEIGHTINGS = ("none", "sqrt_inverse", "balanced")
_POOLINGS = ("arithmetic", "geometric")
_MODEL_COUNT = 7
_LOG_FLOOR = 1e-12
_THREAD_LIMIT = 4
_MIN_GPU_FREE_MIB = 1024
_NORMALIZATION_ATOL = 1e-8
_VALIDATION_EVIDENCE = "CACHED_BASE_OOF_META_CV_ADAPTIVE_NOT_NESTED"


def declarations() -> list[dict]:
    """Return the complete, deterministically ordered 54-item search grid."""

    return [
        {
            "representation": _SEARCH_REPRESENTATION,
            "depth": depth,
            "iterations": iterations,
            "l2_leaf_reg": l2_leaf_reg,
            "weighting": weighting,
        }
        for depth in _DEPTHS
        for iterations in _ITERATIONS
        for l2_leaf_reg in _L2_LEAF_REGS
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
    required = {"representation", "depth", "iterations", "l2_leaf_reg", "weighting"}
    if set(declaration) != required:
        raise ValueError(
            "declaration must contain exactly representation, depth, iterations, "
            "l2_leaf_reg, and weighting"
        )
    representation = declaration["representation"]
    weighting = declaration["weighting"]
    if representation != _SEARCH_REPRESENTATION:
        raise ValueError(f"unknown CatBoost representation: {representation!r}")
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"unknown weighting: {weighting!r}")
    for key in ("depth", "iterations"):
        value = declaration[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{key} must be a positive integer")
        if int(value) <= 0:
            raise ValueError(f"{key} must be positive")
    l2_leaf_reg = declaration["l2_leaf_reg"]
    if isinstance(l2_leaf_reg, (bool, np.bool_)) or not isinstance(l2_leaf_reg, Real):
        raise TypeError("l2_leaf_reg must be a finite positive real number")
    if not np.isfinite(l2_leaf_reg) or float(l2_leaf_reg) <= 0.0:
        raise ValueError("l2_leaf_reg must be finite and positive")
    return copy.deepcopy(declaration)


def _load_catboost():
    """Lazy import keeps cache-only replay and CPU inference independent of GPU discovery."""

    try:
        from catboost import CatBoostClassifier
        from catboost.utils import get_gpu_device_count
    except ImportError as exc:
        raise RuntimeError("CatBoost is required for a new GPU fit; CPU fallback is forbidden") from exc
    return CatBoostClassifier, get_gpu_device_count


def _query_gpu0() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("GPU0 admission failed: nvidia-smi is unavailable") from exc
    matches = []
    for line in output.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6:
            raise RuntimeError("GPU0 admission failed: malformed nvidia-smi output")
        try:
            row = {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "driver_version": values[3],
                "memory_total_mib": int(values[4]),
                "memory_free_mib": int(values[5]),
            }
        except ValueError as exc:
            raise RuntimeError("GPU0 admission failed: malformed numeric device fields") from exc
        if row["index"] == 0:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError("GPU0 admission failed: device index 0 is not uniquely available")
    return matches[0]


def _admit_gpu0(get_gpu_device_count) -> dict:
    try:
        accessible = int(get_gpu_device_count())
    except Exception as exc:
        raise RuntimeError("GPU0 admission failed: CatBoost device discovery failed") from exc
    if accessible < 1:
        raise RuntimeError("GPU0 admission failed: CatBoost cannot access GPU0")
    device = _query_gpu0()
    if device["memory_total_mib"] <= 0 or device["memory_free_mib"] < _MIN_GPU_FREE_MIB:
        raise RuntimeError(
            "GPU0 admission failed: insufficient free memory "
            f"({device['memory_free_mib']} MiB < {_MIN_GPU_FREE_MIB} MiB)"
        )
    return device


def _requested_parameters(declaration: dict, class_count: int) -> dict:
    return {
        "loss_function": "MultiClass",
        "iterations": int(declaration["iterations"]),
        "depth": int(declaration["depth"]),
        "learning_rate": 0.05,
        "l2_leaf_reg": float(declaration["l2_leaf_reg"]),
        "task_type": "GPU",
        "devices": "0",
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "border_count": 64,
        "random_seed": 42,
        "thread_count": _THREAD_LIMIT,
        "verbose": False,
        "allow_writing_files": False,
        "use_best_model": False,
        "gpu_ram_part": 0.65,
        "classes_count": int(class_count),
    }


def _same_parameter(actual, expected) -> bool:
    if isinstance(expected, float):
        try:
            # CatBoost may serialize configured floats through float32 precision.
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=2e-8))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _feature_count_from_model(model) -> tuple[int, dict]:
    raw_count = getattr(model, "n_features_in_", None)
    count = int(raw_count) if isinstance(raw_count, Integral) and not isinstance(raw_count, bool) else 0
    names = getattr(model, "feature_names_", None)
    names_count = len(names) if isinstance(names, (list, tuple)) else 0
    if count > 0 and names_count > 0 and count != names_count:
        raise ValueError("saved CatBoost feature metadata is inconsistent")
    resolved = count if count > 0 else names_count
    if resolved <= 0:
        raise ValueError("saved CatBoost feature count is not positive")
    return resolved, {"n_features_in": count, "feature_names_count": names_count}


def _validate_catboost_model(model, declaration, class_count, feature_count, tree_count=None) -> dict:
    requested = _requested_parameters(declaration, class_count)
    try:
        configured = model.get_params()
        effective = model.get_all_params()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("saved CatBoost parameters are unavailable") from exc
    for key, expected in requested.items():
        if key not in configured or not _same_parameter(configured[key], expected):
            raise ValueError(f"saved CatBoost requested parameter mismatch: {key}")
    effective_required = {
        "task_type": "GPU",
        "loss_function": "MultiClass",
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "learning_rate": 0.05,
        "subsample": 0.8,
        "devices": "0",
        "gpu_ram_part": 0.65,
        "use_best_model": False,
        "classes_count": int(class_count),
        "iterations": requested["iterations"],
        "depth": requested["depth"],
        "l2_leaf_reg": requested["l2_leaf_reg"],
        "border_count": 64,
        "random_seed": 42,
    }
    # CatBoost 1.2.8 omits thread_count, verbose, and allow_writing_files from
    # get_all_params(); those remain mandatory in the exact constructor snapshot above.
    for key, expected in effective_required.items():
        if key not in effective or not _same_parameter(effective[key], expected):
            raise ValueError(f"saved CatBoost effective parameter mismatch: {key}")
    actual_tree_count = getattr(model, "tree_count_", None)
    if not isinstance(actual_tree_count, Integral) or int(actual_tree_count) != requested["iterations"]:
        raise ValueError("saved CatBoost tree_count does not match requested iterations")
    if tree_count is not None and int(actual_tree_count) != int(tree_count):
        raise ValueError("saved CatBoost tree_count does not match artifact receipt")
    resolved_features, feature_sources = _feature_count_from_model(model)
    if resolved_features != int(feature_count):
        raise ValueError("saved CatBoost feature count does not match artifact")
    model_classes = np.asarray(getattr(model, "classes_", None))
    expected_classes = np.arange(class_count, dtype=np.int64)
    if model_classes.ndim != 1 or model_classes.dtype.kind not in "iu" or not np.array_equal(
        model_classes.astype(np.int64, copy=False), expected_classes
    ):
        raise ValueError("saved CatBoost classes do not match exact integer class order")
    return {
        "requested_parameters": requested,
        "effective_parameters": copy.deepcopy(effective),
        "tree_count": int(actual_tree_count),
        "feature_count": resolved_features,
        "feature_count_sources": feature_sources,
        "class_count": int(class_count),
    }


def fit_meta(declaration: dict, p, y) -> dict:
    """Fit one declared train-only CatBoost GPU meta learner without CPU fallback."""

    validated_declaration = _validated_declaration(declaration)
    probability = _as_probability_tensor(p)
    class_count = probability.shape[2]
    labels, _ = _validated_labels(y, class_count)
    if labels.size != probability.shape[1]:
        raise ValueError("y length must match the number of probability rows")
    matrix = features(probability, validated_declaration["representation"])
    weights = sample_weights(labels, validated_declaration["weighting"], class_count)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    CatBoostClassifier, get_gpu_device_count = _load_catboost()
    gpu_admission = _admit_gpu0(get_gpu_device_count)
    requested_parameters = _requested_parameters(validated_declaration, class_count)
    model = CatBoostClassifier(**requested_parameters)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(scaled, labels, sample_weight=weights)
    expected_classes = np.arange(class_count, dtype=np.int64)
    validation = _validate_catboost_model(
        model, validated_declaration, class_count, matrix.shape[1]
    )

    effective_parameters = {
        "execution": "GPU_REQUIRED_NO_CPU_FALLBACK",
        **validation,
        "gpu_admission": gpu_admission,
        "inference": {"device": "CPU_ALLOWED", "thread_count": _THREAD_LIMIT},
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
        "convergence_warnings": [str(item.message) for item in caught_warnings],
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

    effective_parameters = artifact["effective_parameters"]
    if not isinstance(effective_parameters, dict):
        raise ValueError("artifact effective_parameters must be a dictionary")
    validation = _validate_catboost_model(
        artifact["model"],
        declaration,
        expected_classes.size,
        matrix.shape[1],
        tree_count=effective_parameters.get("tree_count"),
    )
    if effective_parameters.get("requested_parameters") != validation["requested_parameters"]:
        raise ValueError("artifact requested CatBoost parameters changed")
    model_classes = np.asarray(artifact["model"].classes_, dtype=np.int64)

    scaled = artifact["scaler"].transform(matrix)
    raw = np.asarray(
        artifact["model"].predict_proba(scaled, thread_count=_THREAD_LIMIT),
        dtype=np.float64,
    )
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
