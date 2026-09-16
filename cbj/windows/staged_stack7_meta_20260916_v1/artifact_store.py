"""Immutable readers and per-fold fit cache for the Stack7 meta search."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np


_INNER_KEYS = (
    "model_ids", "class_order", "inner_ids", "inner_y", "inner_groups",
    "inner_outer_folds", "inner_folds", "inner_probability",
)
_FULL_KEYS = (
    "model_ids", "class_order", "full_ids", "full_y", "full_groups",
    "full_outer_folds", "full_oof_probability",
)
_IDENTITY_KEYS = {
    "source_sha256", "runtime_config_sha256", "bank_sha256", "recipe_sha256",
    "declaration_sha256", "fit_ids_sha256", "fit_y_sha256", "fit_groups_sha256",
    "fit_folds_sha256", "valid_ids_sha256", "valid_y_sha256", "valid_groups_sha256",
    "valid_folds_sha256", "model_order", "class_order", "inner_fold",
}
_DEPLOYMENT_IDENTITY_KEYS = (_IDENTITY_KEYS - {"inner_fold"}) | {
    "schema_version", "feature_role", "fold",
    "fit_probability_tensor_sha256", "valid_probability_tensor_sha256",
}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def canonical_sha256(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_sha256(value):
    return canonical_sha256(np.asarray(value).tolist())


def probability_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _write_atomic_once(path, payload):
    path = Path(path)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact changed: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.is_file():
        if temporary.read_bytes() != payload:
            raise ValueError(f"foreign partial retained: {temporary.name}")
    else:
        with temporary.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bank_config(config):
    required = {
        "training_bank_npz", "training_bank_receipt_json", "training_bank_sha256",
        "training_bank_receipt_sha256", "bank_preparation_source_sha256",
        "bank_preparation_config_sha256", "expected_model_ids", "expected_inner_rows",
        "expected_full_rows", "expected_classes",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"bank config missing keys: {sorted(missing)}")
    return Path(config["training_bank_npz"]), Path(config["training_bank_receipt_json"])


def _verify_bank_receipt(config):
    bank_path, receipt_path = _bank_config(config)
    if file_sha256(bank_path) != config["training_bank_sha256"]:
        raise ValueError("training bank hash mismatch")
    if file_sha256(receipt_path) != config["training_bank_receipt_sha256"]:
        raise ValueError("training bank receipt hash mismatch")
    receipt = _read_json(receipt_path)
    identity = receipt.get("identity", {})
    if receipt.get("schema_version") != "STACK7_TRAINING_BANK_V2" or receipt.get("status") != "COMPLETE":
        raise ValueError("training bank receipt is not complete")
    if receipt.get("bank_sha256") != config["training_bank_sha256"]:
        raise ValueError("training bank receipt does not bind bank hash")
    if identity.get("meta_source_sha256") != config["bank_preparation_source_sha256"]:
        raise ValueError("bank preparation source binding mismatch")
    if identity.get("runtime_config_sha256") != config["bank_preparation_config_sha256"]:
        raise ValueError("bank preparation config binding mismatch")
    expected_models = list(config["expected_model_ids"])
    if len(expected_models) != 7 or len(set(expected_models)) != 7:
        raise ValueError("expected model order must contain exactly seven unique heads")
    if receipt.get("model_order") != expected_models:
        raise ValueError("training bank model order binding mismatch")
    if receipt.get("inner_shape") != [7, int(config["expected_inner_rows"]), int(config["expected_classes"])]:
        raise ValueError("training bank inner dimensions mismatch")
    if receipt.get("full_shape") != [7, int(config["expected_full_rows"]), int(config["expected_classes"])]:
        raise ValueError("training bank full dimensions mismatch")
    if receipt.get("test_probability_read") is not False or receipt.get("test_data_read") is not False:
        raise ValueError("training bank receipt reports forbidden test access")
    return bank_path, receipt_path, receipt


def _load_bank_members(config, keys, *, inner):
    bank_path, receipt_path, receipt = _verify_bank_receipt(config)
    if not set(keys).issubset(set(receipt.get("keys", []))):
        raise ValueError("training bank required fields are absent")
    values = {}
    try:
        with np.load(bank_path, allow_pickle=False) as stored:
            if not set(keys).issubset(set(stored.files)):
                raise ValueError("training bank required fields are absent")
            for key in keys:
                values[key] = np.asarray(stored[key])
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("training bank selected members are invalid") from exc
    for key, value in values.items():
        expected = receipt.get("array_sha256", {}).get(key)
        actual = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"training bank array hash mismatch: {key}")
    models = values["model_ids"].astype(str).tolist()
    classes = values["class_order"].astype(str).tolist()
    if models != list(config["expected_model_ids"]) or len(classes) != int(config["expected_classes"]):
        raise ValueError("training bank model/class binding mismatch")
    if len(set(classes)) != len(classes):
        raise ValueError("training bank class order is duplicated")
    prefix = "inner" if inner else "full"
    rows = int(config["expected_inner_rows"] if inner else config["expected_full_rows"])
    ids = values[f"{prefix}_ids"]
    y = values[f"{prefix}_y"]
    groups = values[f"{prefix}_groups"]
    folds_key = "inner_folds" if inner else "full_outer_folds"
    folds = values[folds_key]
    probability_key = "inner_probability" if inner else "full_oof_probability"
    probability = np.asarray(values[probability_key], dtype=np.float64)
    if (ids.shape != (rows,) or y.shape != (rows,) or groups.shape != (rows,)
            or folds.shape != (rows,) or probability.shape != (7, rows, len(classes))):
        raise ValueError("training bank row/class dimensions are invalid")
    if len(set(ids.astype(str).tolist())) != rows:
        raise ValueError("training bank IDs are duplicated")
    if y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= len(classes)):
        raise ValueError("training bank labels are invalid")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("training bank probability values are invalid")
    if not np.allclose(probability.sum(axis=2), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("training bank probabilities are not normalized")
    group_fold = {}
    for group, fold in zip(groups.astype(str), folds):
        if group_fold.setdefault(group, int(fold)) != int(fold):
            raise ValueError("training bank group crosses folds")
    if inner:
        if set(np.unique(folds).tolist()) != {0, 1, 2}:
            raise ValueError("training bank must contain exactly three inner folds")
        for fold in (0, 1, 2):
            train_y = y[folds != fold]
            if set(np.unique(train_y).tolist()) != set(range(len(classes))):
                raise ValueError("an inner training partition lacks a class")
    values["lineage"] = {
        "bank_sha256": config["training_bank_sha256"],
        "bank_receipt_sha256": config["training_bank_receipt_sha256"],
        "bank_preparation_source_sha256": config["bank_preparation_source_sha256"],
        "bank_preparation_config_sha256": config["bank_preparation_config_sha256"],
        "bank_path": str(bank_path), "receipt_path": str(receipt_path),
    }
    return values


def load_inner_bank(config: dict) -> dict:
    """Load and validate only train-only inner members of the bank."""
    return _load_bank_members(config, _INNER_KEYS, inner=True)


def load_full_bank(config: dict) -> dict:
    """Load full OOF members for post-freeze deployment consumers only."""
    return _load_bank_members(config, _FULL_KEYS, inner=False)


def _validate_fit_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("fit identity must be a dictionary")
    deployment = identity.get("schema_version") == "STACK7_META_DEPLOYMENT_FIT_IDENTITY_V1"
    expected_keys = _DEPLOYMENT_IDENTITY_KEYS if deployment else _IDENTITY_KEYS
    if set(identity) != expected_keys:
        raise ValueError(f"fit identity must contain exactly: {sorted(expected_keys)}")
    if len(identity["model_order"]) != 7 or len(set(identity["model_order"])) != 7:
        raise ValueError("fit identity model order is invalid")
    if len(identity["class_order"]) < 2 or len(set(identity["class_order"])) != len(identity["class_order"]):
        raise ValueError("fit identity class order is invalid")
    if deployment:
        role, fold = identity["feature_role"], identity["fold"]
        if (role == "full_meta_outer_cv" and fold not in (0, 1, 2, 3, 4)) or (
                role == "outer0_confirmation_inner_oof" and fold != 0) or role not in {
                    "full_meta_outer_cv", "outer0_confirmation_inner_oof",
                }:
            raise ValueError("deployment fit identity role/fold is invalid")
        for key in ("fit_probability_tensor_sha256", "valid_probability_tensor_sha256"):
            value = identity[key]
            try:
                valid = isinstance(value, str) and len(value) == 64 and int(value, 16) >= 0
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"deployment fit identity hash is invalid: {key}")
    elif identity["inner_fold"] not in (0, 1, 2):
        raise ValueError("fit identity fold is invalid")
    canonical_sha256(identity)


def _validate_probability(value, classes):
    try:
        probability = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("valid probability is not numeric") from exc
    if (probability.ndim != 2 or probability.shape[0] < 1 or probability.shape[1] != classes
            or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)):
        raise ValueError("valid probability is invalid")
    return probability


def _load_complete(target, identity, receipt):
    model_path = target / "model.joblib"
    probability_path = target / "probability.npz"
    if receipt.get("identity") != identity or receipt.get("identity_sha256") != canonical_sha256(identity):
        raise ValueError("fit cache identity mismatch")
    if not model_path.is_file() or file_sha256(model_path) != receipt.get("model_sha256"):
        raise ValueError("fit cache model hash mismatch")
    if not probability_path.is_file() or file_sha256(probability_path) != receipt.get("probability_npz_sha256"):
        raise ValueError("fit cache probability hash mismatch")
    try:
        with np.load(probability_path, allow_pickle=False) as stored:
            if stored.files != ["probability"]:
                raise ValueError("fit cache probability members changed")
            probability = _validate_probability(stored["probability"], len(identity["class_order"]))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("fit cache probability artifact is invalid") from exc
    if probability_sha256(probability) != receipt.get("valid_probability_sha256"):
        raise ValueError("fit cache probability member hash mismatch")
    # The complete receipt and byte hashes are verified before deserialization.
    model = joblib.load(model_path)
    returned = dict(receipt)
    returned["model_path"] = str(model_path)
    returned["probability_path"] = str(probability_path)
    return model, probability, returned


def load_or_fit_head(root, identity: dict, fit_callback, valid_probability_callback) -> tuple:
    """Reuse or atomically publish one identity-bound fold model and OOF probability."""
    _validate_fit_identity(identity)
    target = Path(root) / canonical_sha256(identity)
    target.mkdir(parents=True, exist_ok=True)
    complete_path = target / "COMPLETE.json"
    pending_path = target / "PENDING.json"
    model_path = target / "model.joblib"
    probability_path = target / "probability.npz"
    model_partial = target / ".model.joblib.partial"
    probability_partial = target / ".probability.npz.partial"
    receipt_partial = target / ".COMPLETE.json.partial"
    if complete_path.is_file():
        return _load_complete(target, identity, _read_json(complete_path))
    partials = [path for path in (model_partial, probability_partial, receipt_partial, model_path, probability_path) if path.exists()]
    if partials and not pending_path.is_file():
        raise ValueError("unbound partial fit artifacts retained")
    if pending_path.is_file():
        pending = _read_json(pending_path)
        if pending.get("identity") != identity or pending.get("identity_sha256") != canonical_sha256(identity):
            raise ValueError("pending fit identity mismatch")
        model_candidate = model_path if model_path.is_file() else model_partial
        probability_candidate = probability_path if probability_path.is_file() else probability_partial
        if (not model_candidate.is_file() or file_sha256(model_candidate) != pending.get("model_sha256")
                or not probability_candidate.is_file() or file_sha256(probability_candidate) != pending.get("probability_npz_sha256")):
            raise ValueError("pending fit artifact hash mismatch")
        if not model_path.is_file(): os.replace(model_partial, model_path)
        if not probability_path.is_file(): os.replace(probability_partial, probability_path)
        receipt = {"schema_version": "STACK7_META_FIT_V1", "status": "COMPLETE", **pending}
        _write_atomic_once(complete_path, _json_bytes(receipt))
        return _load_complete(target, identity, receipt)

    model = fit_callback()
    probability = _validate_probability(valid_probability_callback(model), len(identity["class_order"]))
    joblib.dump(model, model_partial, compress=3)
    with probability_partial.open("wb") as stream:
        np.savez_compressed(stream, probability=probability); stream.flush(); os.fsync(stream.fileno())
    pending = {
        "identity": identity, "identity_sha256": canonical_sha256(identity),
        "model_sha256": file_sha256(model_partial),
        "probability_npz_sha256": file_sha256(probability_partial),
        "valid_probability_sha256": probability_sha256(probability),
        "rows": int(probability.shape[0]), "classes": int(probability.shape[1]),
    }
    _write_atomic_once(pending_path, _json_bytes(pending))
    os.replace(model_partial, model_path); os.replace(probability_partial, probability_path)
    receipt = {"schema_version": "STACK7_META_FIT_V1", "status": "COMPLETE", **pending}
    _write_atomic_once(complete_path, _json_bytes(receipt))
    return _load_complete(target, identity, receipt)
