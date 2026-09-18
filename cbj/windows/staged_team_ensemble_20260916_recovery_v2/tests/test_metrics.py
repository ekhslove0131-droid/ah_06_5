from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import assemble_final_oof, load_final_oof, milestone_status, score_final_oof


class MetricsTests(unittest.TestCase):
    def test_exact_once_assembly_save_readback_and_terminal_score(self):
        ids = np.asarray([f"I{i}" for i in range(10)])
        y = np.asarray([i % 2 for i in range(10)])
        groups = np.asarray([f"G{i}" for i in range(10)])
        folds = np.asarray([i % 5 for i in range(10)])
        results = []
        for fold in range(5):
            indices = np.flatnonzero(folds == fold)
            probability = np.full((len(indices), 2), 0.1)
            probability[np.arange(len(indices)), y[indices]] = 0.9
            results.append({"fold": fold, "valid_indices": indices, "probability": probability})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_PROCEDURE_OOF.npz"
            receipt = assemble_final_oof(results, ids, y, groups, folds, ["A", "B"], path)
            self.assertEqual(len(receipt["fold_receipts"]), 5)
            replay = load_final_oof(path, receipt)
            self.assertEqual(score_final_oof(replay)["macro_f1"], 1.0)
            np.testing.assert_array_equal(replay["ids"], ids)
            np.testing.assert_array_equal(replay["coverage"], np.ones(10, dtype=np.uint8))

    def test_duplicate_or_missing_outer_rows_fail_before_scoring(self):
        ids = np.asarray(["I0", "I1"]); y = np.asarray([0, 1])
        groups = np.asarray(["G0", "G1"]); folds = np.asarray([0, 1])
        result = {"fold": 0, "valid_indices": np.asarray([0, 0]), "probability": np.asarray([[.9, .1], [.9, .1]])}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "coverage"):
                assemble_final_oof([result], ids, y, groups, folds, ["A", "B"], Path(directory) / "x.npz")

    def test_milestone_status_never_authorizes_next_bundle_or_submission(self):
        reached = milestone_status(0.6)
        review = milestone_status(0.599999)
        self.assertEqual(reached["status"], "TARGET_REACHED")
        self.assertEqual(review["status"], "BUNDLE_COMPLETE_REVIEW")
        self.assertFalse(reached["automatic_next_bundle"])
        self.assertFalse(reached["submission"])


if __name__ == "__main__":
    unittest.main()
