"""Sequential exact-once base bank with immutable fold probability reuse."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from base_receipts import canonical_sha256, sequence_sha256, validate_oof
from cache_store import make_cache_key


def _lookup_key(prebindings):
    return hashlib.sha256((json.dumps(prebindings, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _validate_declarations(declarations):
    ids = [row["candidate_id"] for row in declarations if row.get("status") == "EXECUTABLE"]
    if len(ids) != len(set(ids)):
        raise ValueError("executable candidate IDs are duplicated")
    return ids


def run_base_bank(
    declarations,
    folds,
    y,
    class_order,
    cache,
    fit_predict,
    *,
    input_sha256,
    parser_sha256,
    source_sha256,
    ids,
    groups,
    partition_sha256,
):
    folds, y, ids, groups = map(np.asarray, (folds, y, ids, groups))
    if any(value.shape != y.shape for value in (folds, ids, groups)):
        raise ValueError("base bank identity arrays are misaligned")
    executable_ids = _validate_declarations(declarations)
    executable = [row for row in declarations if row.get("status") == "EXECUTABLE"]
    if set(np.unique(folds).tolist()) != set(range(3)):
        raise ValueError("base bank miniature/inner folds must be exactly 0,1,2")
    rows, classes = len(y), len(class_order)
    probabilities = {candidate_id: np.zeros((rows, classes), dtype=np.float64) for candidate_id in executable_ids}
    coverage = np.zeros((len(executable), rows), dtype=np.uint8)
    reused = 0

    # Registration is completed before the first model callback is invoked.
    registered_candidates = list(executable_ids)
    for candidate_index, declaration in enumerate(executable):
        for fold in range(3):
            valid_indices = np.flatnonzero(folds == fold)
            train_indices = np.flatnonzero(folds != fold)
            prebindings = {
                "schema_version": "BASE_BANK_LOOKUP_V1",
                "candidate_id": declaration["candidate_id"],
                "declaration_sha256": declaration["config_hash"],
                "fold": fold,
                "input_sha256": input_sha256,
                "parser_sha256": parser_sha256,
                "source_sha256": source_sha256,
                "fit_ids_sha256": sequence_sha256(ids[train_indices]),
                "fit_y_sha256": sequence_sha256(y[train_indices]),
                "fit_groups_sha256": sequence_sha256(groups[train_indices]),
                "partition_sha256": canonical_sha256([partition_sha256, fold, valid_indices.tolist()]),
                "parameters_sha256": declaration["config_hash"],
            }
            lookup = _lookup_key(prebindings)
            full_key = cache.resolve_index(lookup)
            fold_probability = None
            if full_key is not None:
                receipt = json.loads((cache.artifact_path(full_key) / "COMPLETE.json").read_text(encoding="utf-8"))
                fold_probability, _ = cache.load_or_adopt(full_key, receipt["bindings"])
                reused += 1
            else:
                fold_probability, fit_receipt = fit_predict(declaration, train_indices, valid_indices)
                feature_hash = fit_receipt.get("feature_order_sha256")
                bindings = {key: value for key, value in prebindings.items() if key.endswith("sha256")}
                bindings["feature_order_sha256"] = feature_hash
                full_key = make_cache_key(bindings, supervised=True)
                fold_probability, _ = cache.write(full_key, fold_probability, bindings)
                cache.bind_index(lookup, full_key)
            if fold_probability.shape != (len(valid_indices), classes):
                raise ValueError("base fold probability shape mismatch")
            probabilities[declaration["candidate_id"]][valid_indices] = fold_probability
            coverage[candidate_index, valid_indices] += 1

    for candidate_index, candidate_id in enumerate(executable_ids):
        validate_oof(probabilities[candidate_id], coverage[candidate_index], rows, classes)
    return {
        "schema_version": "BASE_BANK_RESULT_V1",
        "registered_candidates": registered_candidates,
        "probabilities": probabilities,
        "coverage": coverage,
        "reused_folds": reused,
        "test_read": False,
    }


def refit_selected(declarations, weights, refit_one):
    executable = {row["candidate_id"]: row for row in declarations if row.get("status") == "EXECUTABLE"}
    unknown = set(weights) - set(executable)
    if unknown:
        raise ValueError("selected weights reference a non-executable candidate")
    artifacts = {}
    not_refit = []
    for candidate_id in executable:
        if float(weights.get(candidate_id, 0.0)) > 0:
            artifacts[candidate_id] = refit_one(executable[candidate_id])
        else:
            not_refit.append(candidate_id)
    return {"artifacts": artifacts, "not_refit": not_refit, "test_read": False}

