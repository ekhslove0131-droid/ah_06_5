"""Inner-only recipe selection and label-blind outer-valid prediction."""

from __future__ import annotations

import hashlib

import numpy as np

from ensemble_grid import pool_probabilities, run_round0
from log_bias import apply_log_bias
from meta_rounds import run_promotion_loop


def select_inner_recipe(anchors, bank_by_slot, inner_y, guard_mask):
    inner_y = np.asarray(inner_y)
    round0 = run_round0(anchors, bank_by_slot, inner_y, guard_mask)
    promotion = run_promotion_loop(round0, inner_y, max_rounds=3)
    return {
        "schema_version": "INNER_SELECTION_RESULT_V1",
        "selected": promotion.best,
        "promotion": promotion,
        "round0_evaluated": len(round0),
        "round0_by_id": {row.recipe_id: row for row in round0},
        "candidate_graph": promotion.all_candidates,
        "outer_valid_labels_accepted": False,
        "test_read": False,
    }


def build_refit_plan(selected, candidate_graph):
    base_ids = set()
    anchor_branches = set()
    visiting = set()

    def visit(candidate):
        if candidate.recipe_id in visiting:
            raise ValueError("selected recipe graph contains a cycle")
        visiting.add(candidate.recipe_id)
        if candidate.round_index == 0:
            anchor_branches.add(candidate.anchor_id)
            if len(candidate.weights) != 5 or len(candidate.slot_candidate_ids) != 4:
                raise ValueError("Round 0 recipe shape is invalid")
            for weight, candidate_id in zip(candidate.weights[1:], candidate.slot_candidate_ids):
                if weight > 0:
                    base_ids.add(candidate_id)
        else:
            if len(candidate.weights) != len(candidate.slot_candidate_ids):
                raise ValueError("meta recipe source shape is invalid")
            for weight, recipe_id in zip(candidate.weights, candidate.slot_candidate_ids):
                if weight > 0:
                    if recipe_id not in candidate_graph:
                        raise ValueError("selected meta recipe source is missing")
                    visit(candidate_graph[recipe_id])
        visiting.remove(candidate.recipe_id)

    visit(selected)
    return {
        "schema_version": "OUTER_REFIT_PLAN_V1",
        "selected_recipe_id": selected.recipe_id,
        "base_candidate_ids": sorted(base_ids),
        "anchor_branches": sorted(anchor_branches),
        "anchor_components": ["H1", "C10"],
        "outer_valid_labels_accepted": False,
    }


def apply_selected_recipe(selected, candidate_graph, anchors, base_probabilities, guard_mask):
    cache = {}
    guard_mask = np.asarray(guard_mask)

    def apply(candidate):
        if candidate.recipe_id in cache:
            return cache[candidate.recipe_id]
        if candidate.round_index == 0:
            if candidate.anchor_id not in anchors:
                raise ValueError("outer anchor prediction is missing")
            anchor = np.asarray(anchors[candidate.anchor_id], dtype=np.float64)
            sources = [anchor]
            for weight, candidate_id in zip(candidate.weights[1:], candidate.slot_candidate_ids):
                if weight > 0 and candidate_id not in base_probabilities:
                    raise ValueError("outer base prediction is missing")
                sources.append(base_probabilities[candidate_id] if weight > 0 else np.full_like(anchor, np.nan))
            probability = pool_probabilities(sources, candidate.weights, candidate.pooling)
            if guard_mask.shape != (len(probability),) or guard_mask.dtype != np.bool_:
                raise ValueError("outer support guard mask is invalid")
            probability[guard_mask] = anchor[guard_mask]
            probability /= probability.sum(axis=1, keepdims=True)
        else:
            sources = []
            shape = next(iter(anchors.values())).shape
            for weight, recipe_id in zip(candidate.weights, candidate.slot_candidate_ids):
                if weight > 0 and recipe_id not in candidate_graph:
                    raise ValueError("outer meta source recipe is missing")
                sources.append(apply(candidate_graph[recipe_id]) if weight > 0 else np.full(shape, np.nan))
            probability = pool_probabilities(sources, candidate.weights, candidate.pooling)
            if candidate.bias:
                probability = apply_log_bias(probability, np.asarray(candidate.bias, dtype=np.float64))
        cache[candidate.recipe_id] = probability
        return probability

    return apply(selected)


def select_then_predict_outer(selection, outer_train_indices, outer_valid_indices, refit_predict):
    train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    valid_indices = np.asarray(outer_valid_indices, dtype=np.int64)
    if train_indices.ndim != 1 or valid_indices.ndim != 1 or set(train_indices.tolist()) & set(valid_indices.tolist()):
        raise ValueError("outer train/valid indices are invalid")
    selected = selection["selected"]
    graph = selection["candidate_graph"]
    plan = build_refit_plan(selected, graph)
    refitted = refit_predict(plan, train_indices, valid_indices)
    if not isinstance(refitted, dict) or set(refitted) != {"anchors", "base", "guard_mask"}:
        raise ValueError("outer refit callback returned an invalid payload")
    probability = np.asarray(apply_selected_recipe(
        selected, graph, refitted["anchors"], refitted["base"], refitted["guard_mask"],
    ), dtype=np.float64)
    expected_classes = selected.probability.shape[1]
    if probability.shape != (len(valid_indices), expected_classes):
        raise ValueError("outer-valid probability shape mismatch")
    if not np.isfinite(probability).all() or (probability < 0).any() or np.any(probability.sum(1) <= 0):
        raise ValueError("outer-valid probability values are invalid")
    probability /= probability.sum(axis=1, keepdims=True)
    digest = hashlib.sha256(np.ascontiguousarray(probability, dtype="<f8").tobytes()).hexdigest()
    return {
        "selected_recipe_id": selected.recipe_id,
        "probability": probability,
        "receipt": {
            "schema_version": "OUTER_FOLD_PREDICTION_RECEIPT_V1",
            "train_rows": len(train_indices),
            "valid_rows": len(valid_indices),
            "probability_sha256": digest,
            "outer_valid_labels_accepted": False,
            "selection_frozen_before_outer_prediction": True,
            "refit_plan": plan,
            "test_read": False,
        },
    }
