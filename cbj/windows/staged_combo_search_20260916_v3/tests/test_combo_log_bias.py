from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from combo_log_bias import apply_log_bias, fit_c10_log_bias, optimizer_receipt


ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT.parent / "staged_team_ensemble_20260916_recovery_v2/log_bias.py"
spec = importlib.util.spec_from_file_location("v2_log_bias_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)


class ComboLogBiasTests(unittest.TestCase):
    def assert_exact(self, probability, y):
        expected_bias, expected_receipt = oracle.fit_c10_log_bias(probability, y)
        actual_bias, actual_receipt = fit_c10_log_bias(probability, y)
        np.testing.assert_array_equal(actual_bias, expected_bias)
        self.assertEqual(actual_receipt, expected_receipt)
        np.testing.assert_array_equal(
            apply_log_bias(probability, actual_bias).argmax(1),
            oracle.apply_log_bias(probability, expected_bias).argmax(1),
        )

    def test_random_imbalanced_ties_zeros_and_extremes_match_v2(self):
        rng = np.random.default_rng(913)
        y = np.tile(np.arange(26), 4)
        raw = rng.gamma(.7, 1, size=(len(y), 26)); self.assert_exact(raw / raw.sum(1, keepdims=True), y)
        tied = np.ones((len(y), 26)); tied[:, ::3] = 0; self.assert_exact(tied / tied.sum(1, keepdims=True), y)
        extreme = np.full((len(y), 26), 1e-300); extreme[np.arange(len(y)), y] = 1
        self.assert_exact(extreme / extreme.sum(1, keepdims=True), y)
        self.assertTrue(optimizer_receipt()["oracle_receipt_compatible"])


if __name__ == "__main__": unittest.main()
