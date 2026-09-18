from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_grid import MetaCandidate, enumerate_simplex, pool_probabilities, rank_round0, run_round0
from slot_selector import select_slot_candidates


def probability(seed, rows=60, classes=3):
    rng = np.random.default_rng(seed)
    value = rng.random((rows, classes))
    return value / value.sum(axis=1, keepdims=True)


class EnsembleGridTests(unittest.TestCase):
    def test_complete_quarter_simplex_has_exactly_70_stable_vectors(self):
        weights = enumerate_simplex(step=0.25, slots=5)
        self.assertEqual(len(weights), 70)
        self.assertEqual(weights[0], (0.0, 0.0, 0.0, 0.0, 1.0))
        self.assertTrue(all(abs(sum(weight) - 1.0) < 1e-12 for weight in weights))
        self.assertEqual(len(set(weights)), 70)

    def test_pooling_formulas_and_zero_weight_input_are_exact(self):
        p1 = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        p2 = np.asarray([[0.2, 0.8], [0.9, 0.1]])
        ignored = np.full_like(p1, np.nan)
        arithmetic = pool_probabilities([p1, p2, ignored], [0.5, 0.5, 0], "arithmetic")
        np.testing.assert_allclose(arithmetic, 0.5 * p1 + 0.5 * p2, atol=1e-15)
        geometric = pool_probabilities([p1, p2, ignored], [0.5, 0.5, 0], "geometric")
        expected = np.sqrt(p1 * p2); expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(geometric, expected, atol=1e-15)

    def test_slot_selection_uses_best_standalone_or_guarded_half_pool_and_id_tie(self):
        y = np.asarray([0, 1, 2] * 20)
        anchor = probability(1)
        same = probability(2)
        candidates = {
            "project": {"Z": same, "A": same},
            "kim": {"K": probability(3)},
            "ahn": {"H": probability(4)},
            "cross": {"X": probability(5)},
        }
        selected = select_slot_candidates(anchor, candidates, y, np.zeros(len(y), dtype=bool))
        self.assertEqual(selected["project"]["candidate_id"], "A")
        self.assertIn(selected["project"]["selection_mode"], {"standalone", "anchor_half_arithmetic_guarded"})
        self.assertNotIn("outer_valid_y", inspect.signature(select_slot_candidates).parameters)

    def test_two_anchors_create_all_280_unique_records_even_for_duplicate_predictions(self):
        y = np.asarray([0, 1, 2] * 20)
        shared = probability(7)
        anchors = {"A1": shared, "A2": shared.copy()}
        bank = {
            "project": {"P": shared.copy()},
            "kim": {"K": shared.copy()},
            "ahn": {"H": shared.copy()},
            "cross": {"X": shared.copy()},
        }
        records = run_round0(anchors, bank, y, np.zeros(len(y), dtype=bool))
        self.assertEqual(len(records), 280)
        self.assertEqual(len({record.recipe_id for record in records}), 280)
        self.assertTrue(all(record.probability.shape == (60, 3) for record in records))
        self.assertTrue(all(np.isfinite(record.macro_f1) for record in records))
        self.assertNotIn("outer_valid_y", inspect.signature(run_round0).parameters)

    def test_round0_ties_prefer_sparse_then_arithmetic_then_lexicographic(self):
        shared = probability(9, rows=3)
        def row(recipe, weights, pooling, ids=("P", "K", "H", "X")):
            return MetaCandidate(recipe, 0, "A1", ids, weights, pooling, shared, 0.5, "f" * 64)
        records = [
            row("dense", (0.25, 0.25, 0.25, 0.25, 0), "arithmetic"),
            row("geom", (0, 0, 0, 0, 1), "geometric"),
            row("arith-z", (0, 0, 0, 0, 1), "arithmetic", ("Z", "K", "H", "X")),
            row("arith-a", (0, 0, 0, 0, 1), "arithmetic", ("A", "K", "H", "X")),
        ]
        ranked = rank_round0(records)
        self.assertEqual([record.recipe_id for record in ranked], ["arith-a", "arith-z", "geom", "dense"])


if __name__ == "__main__":
    unittest.main()
