from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from outer_procedure import select_inner_recipe, select_then_predict_outer
from ensemble_grid import MetaCandidate


def probability(seed, rows=60, classes=3):
    rng = np.random.default_rng(seed)
    value = rng.random((rows, classes))
    return value / value.sum(axis=1, keepdims=True)


class OuterProcedureTests(unittest.TestCase):
    def test_outer_labels_cannot_enter_selection_or_refit_interface(self):
        inner_y = np.asarray([index % 3 for index in range(60)])
        anchors = {"A1": probability(1), "A2": probability(2)}
        bank = {
            "project": {"P": probability(3)}, "kim": {"K": probability(4)},
            "ahn": {"H": probability(5)}, "cross": {"X": probability(6)},
        }
        selection = select_inner_recipe(anchors, bank, inner_y, np.zeros(60, dtype=bool))
        calls = []
        def refit_predict(plan, train_indices, valid_indices):
            calls.append((plan, tuple(train_indices), tuple(valid_indices)))
            self.assertEqual(plan["anchor_components"], ["H1", "C10"])
            return {
                "anchors": {plan["anchor_branches"][0]: probability(7, rows=len(valid_indices))},
                "base": {candidate_id: probability(8 + index, rows=len(valid_indices))
                         for index, candidate_id in enumerate(plan["base_candidate_ids"])},
                "guard_mask": np.zeros(len(valid_indices), dtype=bool),
            }
        first = select_then_predict_outer(
            selection, np.asarray([0, 1, 2]), np.asarray([3, 4]), refit_predict,
        )
        # Held-out labels are intentionally created and changed, but neither API accepts them.
        outer_valid_y = np.asarray([0, 1]); outer_valid_y[:] = outer_valid_y[::-1]
        second = select_then_predict_outer(
            selection, np.asarray([0, 1, 2]), np.asarray([3, 4]), refit_predict,
        )
        self.assertEqual(first["selected_recipe_id"], second["selected_recipe_id"])
        np.testing.assert_array_equal(first["probability"], second["probability"])
        self.assertNotIn("outer_valid_y", inspect.signature(select_inner_recipe).parameters)
        self.assertNotIn("outer_valid_y", inspect.signature(select_then_predict_outer).parameters)
        self.assertFalse(first["receipt"]["outer_valid_labels_accepted"])
        self.assertEqual(len(calls), 2)

    def test_inner_selection_returns_complete_round_manifests(self):
        y = np.asarray([index % 3 for index in range(60)])
        anchors = {"A1": probability(10), "A2": probability(11)}
        bank = {
            "project": {"P": probability(12)}, "kim": {"K": probability(13)},
            "ahn": {"H": probability(14)}, "cross": {"X": probability(15)},
        }
        result = select_inner_recipe(anchors, bank, y, np.zeros(60, dtype=bool))
        self.assertEqual(result["round0_evaluated"], 280)
        self.assertTrue(all(row["complete"] for row in result["promotion"].round_manifests))
        self.assertFalse(result["test_read"])

    def test_outer_refit_applies_frozen_positive_weights_and_guard_without_adjustment(self):
        anchor = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        base = np.asarray([[0.2, 0.8], [0.9, 0.1]])
        recipe = MetaCandidate(
            "R0-fixed", 0, "A1", ("P", "K", "H", "X"),
            (0.5, 0.5, 0, 0, 0), "arithmetic", anchor, 0.6, "f" * 64,
        )
        selection = {"selected": recipe, "candidate_graph": {recipe.recipe_id: recipe}}
        def refit_predict(plan, train_indices, valid_indices):
            self.assertEqual(plan["base_candidate_ids"], ["P"])
            return {"anchors": {"A1": anchor}, "base": {"P": base},
                    "guard_mask": np.asarray([True, False])}
        output = select_then_predict_outer(selection, np.asarray([2, 3]), np.asarray([0, 1]), refit_predict)
        expected = (anchor + base) / 2
        expected[0] = anchor[0]
        np.testing.assert_allclose(output["probability"], expected, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
