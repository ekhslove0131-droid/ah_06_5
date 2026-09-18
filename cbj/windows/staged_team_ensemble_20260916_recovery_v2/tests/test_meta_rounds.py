from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_grid import MetaCandidate
from meta_rounds import dynamic_simplex, run_meta_round, run_promotion_loop


def candidate(name, probability, score=0.6, round_index=0):
    return MetaCandidate(name, round_index, "A", (), (1.0,), "arithmetic", probability, score, name.ljust(64, "0")[:64])


def probabilities(seed, rows=90, classes=3):
    rng = np.random.default_rng(seed)
    value = rng.random((rows, classes))
    return value / value.sum(axis=1, keepdims=True)


class MetaRoundTests(unittest.TestCase):
    def test_dynamic_simplex_count_matches_composition_formula(self):
        for slots in (2, 3, 8):
            weights = dynamic_simplex(slots)
            self.assertEqual(len(weights), math.comb(slots + 3, 4))
            self.assertTrue(all(abs(sum(row) - 1) < 1e-12 for row in weights))

    def test_one_round_evaluates_two_pooling_and_five_bias_strengths_per_vector(self):
        y = np.asarray([index % 3 for index in range(90)])
        retained = [candidate("A", probabilities(1)), candidate("B", probabilities(2))]
        rows, manifest = run_meta_round(retained, y, round_index=1)
        self.assertEqual(len(rows), math.comb(5, 4) * 2 * 5)
        self.assertEqual(manifest["evaluated"], 50)
        self.assertTrue(manifest["complete"])
        zero = [row for row in rows if row.bias_strength == 0]
        self.assertEqual(len(zero), 10)
        for row in zero:
            self.assertTrue(all(abs(value) < 1e-15 for value in row.bias))

    def test_loop_stops_after_complete_round_for_too_few_promotions(self):
        y = np.asarray([index % 3 for index in range(90)])
        first = candidate("only", probabilities(3), score=0.56)
        below = candidate("below", probabilities(4), score=0.54)
        result = run_promotion_loop([first, below], y, max_rounds=3)
        self.assertEqual(result.stop_reason, "FEWER_THAN_TWO_PROMOTED")
        self.assertEqual(len(result.round_manifests), 1)
        self.assertTrue(result.round_manifests[0]["complete"])

    def test_loop_never_runs_more_than_three_meta_rounds(self):
        y = np.asarray([index % 3 for index in range(90)])
        # An injectable builder isolates the stopping-state machine from the
        # exhaustive formula test above and forces strict improvement.
        initial = [candidate("A", probabilities(5), 0.7), candidate("B", probabilities(6), 0.69)]
        def builder(retained, labels, round_index):
            rows = [
                candidate(f"R{round_index}A", probabilities(10 + round_index), 0.7 + round_index * 0.01, round_index),
                candidate(f"R{round_index}B", probabilities(20 + round_index), 0.69 + round_index * 0.01, round_index),
            ]
            return rows, {"round": round_index, "evaluated": 2, "complete": True}
        result = run_promotion_loop(initial, y, max_rounds=3, round_builder=builder)
        self.assertEqual(result.stop_reason, "MAX_ROUNDS_COMPLETE")
        self.assertEqual([row["round"] for row in result.round_manifests], [0, 1, 2, 3])

    def test_loop_stops_only_after_equal_score_round_is_complete(self):
        y = np.asarray([index % 3 for index in range(90)])
        initial = [candidate("A", probabilities(30), 0.7), candidate("B", probabilities(31), 0.69)]
        def builder(retained, labels, round_index):
            rows = [
                candidate("equal", probabilities(32), 0.7, round_index),
                candidate("other", probabilities(33), 0.68, round_index),
            ]
            return rows, {"round": round_index, "evaluated": 2, "complete": True}
        result = run_promotion_loop(initial, y, max_rounds=3, round_builder=builder)
        self.assertEqual(result.stop_reason, "NO_STRICT_IMPROVEMENT")
        self.assertEqual(len(result.round_manifests), 2)
        self.assertTrue(result.round_manifests[-1]["complete"])


if __name__ == "__main__":
    unittest.main()
