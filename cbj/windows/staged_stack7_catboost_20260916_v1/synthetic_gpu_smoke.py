#!/usr/bin/env python3
"""Synthetic-only two-fit CatBoost GPU smoke; no competition-data path exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np

try:
    from .meta_core import fit_meta, predict_meta
except ImportError:
    from meta_core import fit_meta, predict_meta


_CLASSES = 26
_SEED = 42
_DECLARATION = {
    "representation": "log_probability_confidence",
    "depth": 3,
    "iterations": 5,
    "l2_leaf_reg": 3,
    "weighting": "none",
}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synthetic_tensor():
    labels = np.repeat(np.arange(_CLASSES, dtype=np.int64), 2)
    probability = np.full((7, labels.size, _CLASSES), 0.2 / (_CLASSES - 1), dtype=np.float64)
    for model_index in range(7):
        primary = 0.8 - model_index * 0.01
        probability[model_index, np.arange(labels.size), labels] = primary
        remainder = (1.0 - primary) / (_CLASSES - 1)
        mask = np.ones((labels.size, _CLASSES), dtype=bool)
        mask[np.arange(labels.size), labels] = False
        probability[model_index][mask] = remainder
    return probability, labels


def run_smoke(output_root):
    """Run two sequential five-tree fits and verify joblib prediction parity."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "SYNTHETIC_GPU_SMOKE.json"
    if receipt_path.exists():
        raise ValueError("synthetic GPU smoke receipt already exists")
    probability, labels = _synthetic_tensor()
    maximum_reload_error = 0.0
    fit_receipts = []
    with tempfile.TemporaryDirectory(prefix="stack7-catboost-smoke-", dir=root) as temporary:
        temporary_root = Path(temporary)
        for fit_index in range(2):
            artifact = fit_meta(_DECLARATION, probability, labels)
            before = predict_meta(artifact, probability)
            model_path = temporary_root / f"fit_{fit_index}.joblib"
            joblib.dump(artifact, model_path, compress=3)
            restored = joblib.load(model_path)
            after = predict_meta(restored, probability)
            error = float(np.max(np.abs(before - after)))
            if error > 1e-12 or not np.array_equal(before.argmax(1), after.argmax(1)):
                raise ValueError("synthetic GPU smoke joblib prediction parity failed")
            maximum_reload_error = max(maximum_reload_error, error)
            effective = artifact["effective_parameters"]
            fit_receipts.append({
                "fit_index": fit_index,
                "artifact_sha256": _sha256(model_path),
                "gpu_admission": effective["gpu_admission"],
                "effective_parameters": effective["effective_parameters"],
                "tree_count": effective["tree_count"],
                "feature_count": effective["feature_count"],
                "class_count": effective["class_count"],
            })
    receipt = {
        "schema_version": "STACK7_CATBOOST_SYNTHETIC_GPU_SMOKE_V1",
        "status": "SYNTHETIC_GPU_SMOKE_COMPLETE",
        "scope": "SYNTHETIC_ONLY_NO_COMPETITION_DATA",
        "fit_count": 2,
        "classes": _CLASSES,
        "rows": int(labels.size),
        "features": 7 * (_CLASSES + 3),
        "iterations": _DECLARATION["iterations"],
        "depth": _DECLARATION["depth"],
        "reload_max_abs_diff": maximum_reload_error,
        "fits": fit_receipts,
        "cpu_fallback": False,
    }
    payload = _json_bytes(receipt)
    partial = root / ".SYNTHETIC_GPU_SMOKE.json.partial"
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, receipt_path)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stack7-catboost-synthetic-gpu-smoke")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.out), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
