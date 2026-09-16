"""Deterministic outer/inner/sub-inner canonical-group partition contract."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from contracts import CLASS_ORDER


class PartitionContractError(ValueError):
    pass


def _sequence_hash(values) -> str:
    payload = json.dumps(np.asarray(values).tolist(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _indices_receipt(indices, ids, y, groups) -> dict:
    indices = np.asarray(indices, dtype=np.int64)
    return {
        "rows": int(len(indices)),
        "indices_sha256": _sequence_hash(indices),
        "ids_sha256": _sequence_hash(ids[indices]),
        "y_sha256": _sequence_hash(y[indices]),
        "groups_sha256": _sequence_hash(groups[indices]),
    }


def _validate_inputs(ids, y, groups, outer_folds, class_order):
    arrays = [np.asarray(value) for value in (ids, y, groups, outer_folds)]
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
        raise PartitionContractError("ID, label, group and fold arrays must be aligned one-dimensional arrays")
    ids, y, groups, outer_folds = arrays
    if len(set(ids.tolist())) != len(ids):
        raise PartitionContractError("ID values must be unique")
    if not isinstance(class_order, list) or len(class_order) != 26 or len(set(class_order)) != 26:
        raise PartitionContractError("class order must contain 26 unique labels")
    observed = set(y.tolist())
    if observed != set(class_order):
        raise PartitionContractError("class labels differ from class order")
    expected_order = CLASS_ORDER if observed == set(CLASS_ORDER) else sorted(observed)
    if class_order != expected_order:
        raise PartitionContractError("class order is not the frozen production order")
    if set(np.unique(outer_folds).tolist()) != set(range(5)):
        raise PartitionContractError("outer folds must be exactly 0 through 4")
    group_to_fold = {}
    for group, fold in zip(groups.tolist(), outer_folds.tolist()):
        previous = group_to_fold.setdefault(str(group), int(fold))
        if previous != int(fold):
            raise PartitionContractError("canonical group crosses outer folds")
    return ids, y, groups, outer_folds


def _require_partition(train_indices, valid_indices, y, groups, class_order, name):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if set(train_indices.tolist()) & set(valid_indices.tolist()):
        raise PartitionContractError(f"{name} train and validation rows overlap")
    if set(groups[train_indices].tolist()) & set(groups[valid_indices].tolist()):
        raise PartitionContractError(f"canonical group crosses {name} boundary")
    if set(y[train_indices].tolist()) != set(class_order) or set(y[valid_indices].tolist()) != set(class_order):
        raise PartitionContractError(f"a class is missing from {name} train or validation")


def _split_three(base_indices, ids, y, groups, seed, class_order, name):
    base_indices = np.asarray(base_indices, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    result = []
    seen_valid = []
    X = np.zeros((len(base_indices), 1), dtype=np.uint8)
    for fold, (local_train, local_valid) in enumerate(splitter.split(X, y[base_indices], groups[base_indices])):
        train_indices = base_indices[local_train]
        valid_indices = base_indices[local_valid]
        _require_partition(train_indices, valid_indices, y, groups, class_order, f"{name} fold {fold}")
        seen_valid.extend(valid_indices.tolist())
        result.append({
            "fold": fold,
            "train_indices": train_indices.tolist(),
            "valid_indices": valid_indices.tolist(),
            "train_receipt": _indices_receipt(train_indices, ids, y, groups),
            "valid_receipt": _indices_receipt(valid_indices, ids, y, groups),
        })
    if sorted(seen_valid) != sorted(base_indices.tolist()):
        raise PartitionContractError(f"{name} validation does not cover its parent exactly once")
    return result


def build_partitions(ids, y, groups, outer_folds, class_order) -> dict:
    ids, y, groups, outer_folds = _validate_inputs(ids, y, groups, outer_folds, class_order)
    outer_rows = []
    all_indices = np.arange(len(ids), dtype=np.int64)
    for outer_fold in range(5):
        valid_indices = all_indices[outer_folds == outer_fold]
        train_indices = all_indices[outer_folds != outer_fold]
        _require_partition(train_indices, valid_indices, y, groups, class_order, f"outer fold {outer_fold}")
        inner_seed = 2026091400 + outer_fold
        inner_rows = _split_three(train_indices, ids, y, groups, inner_seed, class_order, f"outer {outer_fold} inner")
        for inner_index, inner_row in enumerate(inner_rows):
            subinner_seed = 2026091500 + 10 * outer_fold + inner_index
            inner_row["subinner_seed"] = subinner_seed
            inner_row["subinner"] = _split_three(
                inner_row["train_indices"], ids, y, groups, subinner_seed, class_order,
                f"outer {outer_fold} inner {inner_index} sub-inner",
            )
        outer_rows.append({
            "fold": outer_fold,
            "inner_seed": inner_seed,
            "train_indices": train_indices.tolist(),
            "valid_indices": valid_indices.tolist(),
            "train_receipt": _indices_receipt(train_indices, ids, y, groups),
            "valid_receipt": _indices_receipt(valid_indices, ids, y, groups),
            "valid_ids_sha256": _sequence_hash(ids[valid_indices]),
            "inner": inner_rows,
        })
    return {
        "schema_version": "TEAM_NESTED_PARTITION_BUNDLE_V1",
        "outer_seed": 43,
        "inner_seed_formula": "2026091400+outer_index",
        "subinner_seed_formula": "2026091500+10*outer_index+inner_index",
        "identity": {
            "ids_sha256": _sequence_hash(ids),
            "y_sha256": _sequence_hash(y),
            "groups_sha256": _sequence_hash(groups),
            "outer_folds_sha256": _sequence_hash(outer_folds),
            "rows": int(len(ids)),
            "class_order": list(class_order),
        },
        "outer": outer_rows,
    }


def verify_partition_bundle(bundle, ids, y, groups, outer_folds, class_order) -> None:
    expected = build_partitions(ids, y, groups, outer_folds, class_order)
    if bundle != expected:
        raise PartitionContractError("partition receipt or assignment differs from deterministic reconstruction")
