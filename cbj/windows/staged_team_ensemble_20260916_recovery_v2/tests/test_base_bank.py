from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base_bank import refit_selected, run_base_bank
from cache_store import ImmutableProbabilityCache


class BaseBankTests(unittest.TestCase):
    def test_miniature_bank_has_exact_oof_coverage_and_resume_without_refit(self):
        declarations = [
            {"candidate_id": "C1", "config_hash": "1" * 64, "status": "EXECUTABLE"},
            {"candidate_id": "C2", "config_hash": "2" * 64, "status": "EXECUTABLE"},
        ]
        folds = np.asarray([0, 1, 2, 0, 1, 2])
        y = np.asarray(["A", "B", "C", "A", "B", "C"])
        calls = []

        def fit_predict(declaration, train_indices, valid_indices):
            calls.append((declaration["candidate_id"], tuple(valid_indices)))
            probability = np.full((len(valid_indices), 3), 0.1)
            for row, index in enumerate(valid_indices):
                probability[row, ["A", "B", "C"].index(y[index])] = 0.8
            return probability, {"feature_order_sha256": "f" * 64}

        with tempfile.TemporaryDirectory() as directory:
            cache = ImmutableProbabilityCache(directory)
            first = run_base_bank(
                declarations, folds, y, ["A", "B", "C"], cache, fit_predict,
                input_sha256="a" * 64, parser_sha256="b" * 64,
                source_sha256="c" * 64, ids=np.asarray([f"I{i}" for i in range(6)]),
                groups=np.asarray([f"G{i}" for i in range(6)]), partition_sha256="d" * 64,
            )
            self.assertEqual(len(calls), 6)
            self.assertTrue(np.all(first["coverage"] == 1))
            self.assertEqual(first["registered_candidates"], ["C1", "C2"])
            second = run_base_bank(
                declarations, folds, y, ["A", "B", "C"], cache, fit_predict,
                input_sha256="a" * 64, parser_sha256="b" * 64,
                source_sha256="c" * 64, ids=np.asarray([f"I{i}" for i in range(6)]),
                groups=np.asarray([f"G{i}" for i in range(6)]), partition_sha256="d" * 64,
            )
            self.assertEqual(len(calls), 6)
            np.testing.assert_array_equal(first["probabilities"]["C1"], second["probabilities"]["C1"])
            self.assertEqual(second["reused_folds"], 6)
            self.assertFalse(second["test_read"])

    def test_outer_refit_only_runs_positive_weight_selected_components(self):
        declarations = [
            {"candidate_id": "C1", "config_hash": "1" * 64, "status": "EXECUTABLE"},
            {"candidate_id": "C2", "config_hash": "2" * 64, "status": "EXECUTABLE"},
            {"candidate_id": "C3", "config_hash": "3" * 64, "status": "BLOCKED"},
        ]
        calls = []
        output = refit_selected(
            declarations, {"C1": 0.0, "C2": 0.75},
            lambda declaration: calls.append(declaration["candidate_id"]) or declaration["candidate_id"],
        )
        self.assertEqual(calls, ["C2"])
        self.assertEqual(output["artifacts"], {"C2": "C2"})
        self.assertEqual(output["not_refit"], ["C1"])


if __name__ == "__main__":
    unittest.main()
