from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from log_bias import apply_log_bias, fit_c10_log_bias


class LogBiasTests(unittest.TestCase):
    def test_optimizer_is_deterministic_centered_bounded_and_improving(self):
        rng = np.random.default_rng(42)
        ninety = 90
        y = np.asarray([index % 3 for index in range(ninety)], dtype=np.int64)
        probability = rng.random((ninety, 3))
        probability[:, 0] *= 2.5
        probability /= probability.sum(axis=1, keepdims=True)
        first_bias, first_receipt = fit_c10_log_bias(probability, y)
        second_bias, second_receipt = fit_c10_log_bias(probability, y)
        np.testing.assert_array_equal(first_bias, second_bias)
        self.assertEqual(first_receipt, second_receipt)
        self.assertAlmostEqual(float(first_bias.mean()), 0.0, places=12)
        self.assertLessEqual(float(np.max(np.abs(first_bias))), 1.5 + 1e-12)
        self.assertGreaterEqual(first_receipt["final"]["objective"], first_receipt["initial"]["objective"])
        adjusted = apply_log_bias(probability, first_bias)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0, atol=1e-12)

    def test_wrong_shapes_or_nonfinite_probabilities_fail(self):
        with self.assertRaises(ValueError):
            fit_c10_log_bias(np.ones((5, 2)), np.asarray([0, 1, 0]))
        probability = np.full((6, 3), 1 / 3); probability[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_c10_log_bias(probability, np.asarray([0, 1, 2, 0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
