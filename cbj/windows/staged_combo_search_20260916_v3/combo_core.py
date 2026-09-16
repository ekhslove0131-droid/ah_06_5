"""Small explicit probability recipe graph used by search and deployment."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def canonical_sha256(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def probability_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def normalize(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not len(value) or value.shape[1] < 2:
        raise ValueError("probability shape is invalid")
    if not np.isfinite(value).all() or (value < 0).any() or np.any(value.sum(1) <= 0):
        raise ValueError("probability values are invalid")
    return value / value.sum(axis=1, keepdims=True)


def pool(probabilities, weights, pooling):
    weights = tuple(float(value) for value in weights)
    if (len(probabilities) != len(weights) or not np.isfinite(weights).all()
            or any(value < 0 for value in weights) or abs(sum(weights) - 1) > 1e-12):
        raise ValueError("pool weights are invalid")
    active = [(normalize(value), weight) for value, weight in zip(probabilities, weights) if weight > 0]
    if not active or len({value.shape for value, _ in active}) != 1:
        raise ValueError("active pool inputs are missing or misaligned")
    if pooling == "arithmetic":
        result = sum(weight * value for value, weight in active)
    elif pooling == "geometric":
        logits = sum(weight * np.log(np.maximum(value, 1e-12)) for value, weight in active)
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
    else:
        raise ValueError("unknown pooling")
    return result / result.sum(axis=1, keepdims=True)


def apply_bias(probability, bias):
    probability = normalize(probability)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (probability.shape[1],) or not np.isfinite(bias).all():
        raise ValueError("bias is invalid or misaligned")
    logits = np.log(np.maximum(probability, 1e-300)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def simplex(step=0.25, slots=6):
    if step != 0.25 or slots < 1:
        raise ValueError("combo grid requires step .25 and positive slots")
    output = []
    def visit(prefix, remaining):
        if len(prefix) == slots - 1:
            output.append(tuple(value / 4 for value in prefix + [remaining])); return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value)
    visit([], 4)
    if len(output) != math.comb(slots + 3, 4):
        raise AssertionError("simplex enumeration mismatch")
    return output


def recipe_id(node):
    return "COMBO-" + canonical_sha256(node)


def evaluate(recipe, nodes, leaves, *, mode, fold_ids=None):
    """Evaluate a frozen graph without labels.

    Bias nodes hold cross-fitted fold biases for ``search`` and a single bias
    for ``deployment``.  Guarding is applied once at the final recipe against
    the frozen stage baseline, after all pooling and bias operations.
    """
    cache = {}
    visiting = set()
    def visit(node_id):
        if node_id in cache:
            return cache[node_id]
        if node_id in visiting:
            raise ValueError("recipe graph contains a cycle")
        if node_id not in nodes:
            raise ValueError("recipe graph node is missing")
        visiting.add(node_id)
        node = nodes[node_id]
        kind = node["kind"]
        if kind == "leaf":
            result = normalize(leaves[node["leaf_id"]])
        elif kind == "mix":
            if len(node.get("sources", [])) != len(node.get("weights", [])):
                raise ValueError("mix sources and weights are misaligned")
            raw_weights = np.asarray(node["weights"], dtype=np.float64)
            if (not np.isfinite(raw_weights).all() or (raw_weights < 0).any()
                    or abs(float(raw_weights.sum()) - 1) > 1e-12):
                raise ValueError("mix weights are invalid")
            active = [(source, float(weight)) for source, weight in zip(node["sources"], node["weights"]) if float(weight) > 0]
            result = pool([visit(source) for source, _ in active], [weight for _, weight in active], node["pooling"])
            if "guard_mask_leaf" in node:
                guard_value = np.asarray(leaves[node["guard_mask_leaf"]])
                guard_baseline = visit(node["guard_baseline_source"])
                if guard_value.dtype != np.bool_ or guard_value.shape != (len(result),):
                    raise ValueError("internal guard evidence is invalid")
                result = result.copy(); result[guard_value] = guard_baseline[guard_value]
                result = normalize(result)
        elif kind == "fixed_bias":
            result = apply_bias(visit(node["source"]), node["bias"])
        elif kind == "crossfit_bias":
            base = visit(node["source"])
            strength = float(node["strength"])
            if not np.isfinite(strength) or strength < 0:
                raise ValueError("bias strength is invalid")
            if mode == "search":
                fold_ids_value = np.asarray(fold_ids)
                if fold_ids_value.shape != (len(base),):
                    raise ValueError("search fold IDs are misaligned")
                biases = node["fold_biases"]
                expected_folds = sorted(np.unique(fold_ids_value).tolist())
                if set(biases) != {str(value) for value in expected_folds}:
                    raise ValueError("crossfit bias fold keys are invalid")
                validated = {str(fold): np.asarray(biases[str(fold)], dtype=np.float64) for fold in expected_folds}
                if any(value.shape != (base.shape[1],) or not np.isfinite(value).all() for value in validated.values()):
                    raise ValueError("crossfit bias is invalid")
                if strength == 0:
                    result = base
                else:
                    result = np.zeros_like(base)
                    for fold in expected_folds:
                        mask = fold_ids_value == fold
                        result[mask] = apply_bias(base[mask], validated[str(fold)] * strength)
            elif mode == "deployment":
                deployment_bias = np.asarray(node["deployment_bias"], dtype=np.float64)
                if deployment_bias.shape != (base.shape[1],) or not np.isfinite(deployment_bias).all():
                    raise ValueError("deployment bias is invalid")
                result = base if strength == 0 else apply_bias(base, deployment_bias * strength)
            else:
                raise ValueError("unknown evaluation mode")
        else:
            raise ValueError("unknown recipe node kind")
        visiting.remove(node_id)
        cache[node_id] = result
        return result
    result = visit(recipe["root"])
    if recipe.get("apply_final_guard", True):
        guard = np.asarray(leaves[recipe["guard_mask_leaf"]])
        baseline = normalize(leaves[recipe["guard_baseline_leaf"]])
        if guard.dtype != np.bool_ or guard.shape != (len(result),) or baseline.shape != result.shape:
            raise ValueError("guard evidence is misaligned")
        result = result.copy(); result[guard] = baseline[guard]
    return normalize(result)


def active_leaf_ids(recipe, nodes):
    output = set()
    seen = set(); visiting = set()
    def visit(node_id):
        if node_id in seen:
            return
        if node_id in visiting:
            raise ValueError("recipe graph contains a cycle")
        if node_id not in nodes:
            raise ValueError("recipe graph node is missing")
        visiting.add(node_id)
        node = nodes[node_id]
        kind = node.get("kind")
        if kind == "leaf":
            output.add(node["leaf_id"])
        elif kind in {"fixed_bias", "crossfit_bias"}:
            visit(node["source"])
        elif kind == "mix":
            if len(node.get("sources", [])) != len(node.get("weights", [])):
                raise ValueError("mix sources and weights are misaligned")
            weights = np.asarray(node["weights"], dtype=np.float64)
            if not np.isfinite(weights).all() or (weights < 0).any() or abs(float(weights.sum()) - 1) > 1e-12:
                raise ValueError("mix weights are invalid")
            for weight, source in zip(weights, node["sources"]):
                if float(weight) > 0:
                    visit(source)
            if "guard_mask_leaf" in node:
                output.add(node["guard_mask_leaf"]); visit(node["guard_baseline_source"])
        else:
            raise ValueError("unknown recipe node kind")
        visiting.remove(node_id); seen.add(node_id)
    visit(recipe["root"])
    if recipe.get("apply_final_guard", True):
        output.add(recipe["guard_baseline_leaf"])
        output.add(recipe["guard_mask_leaf"])
    return sorted(output)


def translate_v2_selection(selection):
    """Translate a frozen V2 MetaCandidate graph into explicit nodes."""
    graph = selection["candidate_graph"]
    nodes = {}
    def leaf(leaf_id):
        node_id = "LEAF-" + canonical_sha256(leaf_id)
        nodes.setdefault(node_id, {"kind": "leaf", "leaf_id": leaf_id})
        return node_id
    def visit(candidate):
        if candidate.recipe_id in nodes:
            return candidate.recipe_id
        if candidate.round_index == 0:
            if len(candidate.weights) != 5 or len(candidate.slot_candidate_ids) != 4:
                raise ValueError("V2 Round0 recipe shape is invalid")
            sources = [leaf(f"anchor:{candidate.anchor_id}")]
            sources.extend(leaf(f"candidate:{value}") for value in candidate.slot_candidate_ids)
            nodes[candidate.recipe_id] = {
                "kind": "mix", "sources": sources,
                "weights": list(candidate.weights), "pooling": candidate.pooling,
                "guard_mask_leaf": "mask:guard", "guard_baseline_source": sources[0],
            }
        else:
            if len(candidate.weights) != len(candidate.slot_candidate_ids):
                raise ValueError("V2 meta recipe shape is invalid")
            active = [(value, float(weight)) for value, weight in zip(candidate.slot_candidate_ids, candidate.weights) if float(weight) > 0]
            sources = [visit(graph[value]) for value, _ in active]
            weights = [weight for _, weight in active]
            mix = {
                "kind": "mix", "sources": sources,
                "weights": weights, "pooling": candidate.pooling,
            }
            mix_id = candidate.recipe_id + "-MIX"
            nodes[mix_id] = mix
            nodes[candidate.recipe_id] = {
                "kind": "fixed_bias", "source": mix_id, "bias": list(candidate.bias),
            }
        return candidate.recipe_id
    root = visit(selection["selected"])
    return {"root": root, "nodes": nodes, "leaf": leaf}
