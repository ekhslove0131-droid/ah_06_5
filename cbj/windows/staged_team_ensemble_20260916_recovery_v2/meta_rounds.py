"""Complete ensemble-of-ensembles and bounded class-bias promotion rounds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from sklearn.metrics import f1_score

from ensemble_grid import MetaCandidate, pool_probabilities
from log_bias import apply_log_bias, fit_c10_log_bias
from promotion import promote


BIAS_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class PromotionResult:
    best: MetaCandidate
    retained: list[MetaCandidate]
    round_manifests: list[dict]
    stop_reason: str
    all_candidates: dict[str, MetaCandidate]


def dynamic_simplex(slots):
    if not isinstance(slots, int) or slots < 1:
        raise ValueError("dynamic simplex slots must be positive")
    output = []
    def visit(prefix, remaining):
        if len(prefix) == slots - 1:
            output.append(tuple(value / 4 for value in prefix + [remaining]))
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value)
    visit([], 4)
    if len(output) != math.comb(slots + 3, 4):
        raise AssertionError("dynamic simplex count mismatch")
    return output


def _probability_sha(probability):
    return hashlib.sha256(np.ascontiguousarray(probability, dtype="<f8").tobytes()).hexdigest()


def _recipe_id(payload):
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return f"R{payload['round']}-" + hashlib.sha256(raw).hexdigest()


def _macro(y, probability):
    return float(f1_score(
        y, probability.argmax(axis=1), labels=np.arange(probability.shape[1]),
        average="macro", zero_division=0,
    ))


def run_meta_round(retained, y, *, round_index):
    if round_index < 1 or len(retained) < 2:
        raise ValueError("meta round requires at least two retained inputs and index >= 1")
    y = np.asarray(y, dtype=np.int64)
    source_ids = tuple(candidate.recipe_id for candidate in retained)
    source_probabilities = [candidate.probability for candidate in retained]
    rows = []
    pooled_count = 0
    for weights in dynamic_simplex(len(retained)):
        for pooling in ("arithmetic", "geometric"):
            pooled = pool_probabilities(source_probabilities, weights, pooling)
            bias, bias_receipt = fit_c10_log_bias(pooled, y)
            pooled_count += 1
            for strength in BIAS_STRENGTHS:
                scaled_bias = bias * strength
                probability = pooled if strength == 0 else apply_log_bias(pooled, scaled_bias)
                probability = np.asarray(probability, dtype=np.float64).copy()
                payload = {
                    "schema_version": "META_RECIPE_V1",
                    "round": round_index,
                    "source_recipe_ids": source_ids,
                    "weights": weights,
                    "pooling": pooling,
                    "bias_strength": strength,
                    "raw_bias": bias.tolist(),
                }
                rows.append(MetaCandidate(
                    recipe_id=_recipe_id(payload), round_index=round_index,
                    anchor_id="META", slot_candidate_ids=source_ids,
                    weights=weights, pooling=pooling, probability=probability,
                    macro_f1=_macro(y, probability), probability_sha256=_probability_sha(probability),
                    bias_strength=strength, bias=tuple(scaled_bias.tolist()),
                ))
    expected = math.comb(len(retained) + 3, 4) * 2 * 5
    if len(rows) != expected:
        raise AssertionError("meta round result count mismatch")
    manifest = {
        "schema_version": "META_ROUND_MANIFEST_V1",
        "round": round_index,
        "retained_inputs": len(retained),
        "simplex_vectors": len(dynamic_simplex(len(retained))),
        "pooled_bias_fits": pooled_count,
        "evaluated": len(rows),
        "complete": True,
        "bias_strengths": list(BIAS_STRENGTHS),
        "test_read": False,
    }
    return rows, manifest


def _best(candidates):
    if not candidates:
        raise ValueError("no meta candidates")
    return sorted(candidates, key=lambda row: (-row.macro_f1, row.recipe_id))[0]


def run_promotion_loop(round0, y, *, max_rounds=3, round_builder=run_meta_round):
    if max_rounds != 3:
        raise ValueError("V1 maximum promotion rounds is exactly 3")
    y = np.asarray(y)
    best = _best(round0)
    all_candidates = {row.recipe_id: row for row in round0}
    retained = promote(round0)
    manifests = [{
        "schema_version": "ROUND0_PROMOTION_MANIFEST_V1",
        "round": 0,
        "evaluated": len(round0),
        "promoted": len(retained),
        "best_macro_f1": best.macro_f1,
        "complete": True,
        "test_read": False,
    }]
    if len(retained) < 2:
        return PromotionResult(best, retained, manifests, "FEWER_THAN_TWO_PROMOTED", all_candidates)
    for round_index in range(1, max_rounds + 1):
        rows, manifest = round_builder(retained, y, round_index=round_index)
        for row in rows:
            if row.recipe_id in all_candidates:
                raise ValueError("meta recipe ID collision")
            all_candidates[row.recipe_id] = row
        if not manifest.get("complete"):
            raise ValueError("incomplete meta round cannot be selected")
        round_best = _best(rows)
        improved = round_best.macro_f1 > best.macro_f1 + 1e-12
        if improved:
            best = round_best
        retained = promote(rows)
        manifest = dict(manifest, promoted=len(retained), best_macro_f1=round_best.macro_f1,
                        strict_improvement=improved)
        manifests.append(manifest)
        if len(retained) < 2:
            return PromotionResult(best, retained, manifests, "FEWER_THAN_TWO_PROMOTED", all_candidates)
        if not improved:
            return PromotionResult(best, retained, manifests, "NO_STRICT_IMPROVEMENT", all_candidates)
    return PromotionResult(best, retained, manifests, "MAX_ROUNDS_COMPLETE", all_candidates)
