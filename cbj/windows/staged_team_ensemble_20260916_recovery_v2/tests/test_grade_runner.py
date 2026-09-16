from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grade_runner import _new_stage_probabilities, _save_component, _write_submission
from model_adapters import AEHeadPredictor


class GradeRunnerTests(unittest.TestCase):
    def probability(self, rows=4, classes=3):
        value = np.arange(1, rows * classes + 1, dtype=np.float64).reshape(rows, classes)
        return value / value.sum(axis=1, keepdims=True)

    def test_component_checkpoint_resumes_and_rejects_partition_change(self):
        with tempfile.TemporaryDirectory() as directory:
            bindings = {
                "component": "C", "declaration_sha256": "a" * 64, "fold": 0,
                "source_sha256": "b" * 64, "train_ids_sha256": "c" * 64,
                "train_y_sha256": "d" * 64, "train_groups_sha256": "e" * 64,
                "valid_ids_sha256": "f" * 64, "test_ids_sha256": "0" * 64,
                "class_order": ["C0", "C1", "C2"],
            }
            valid = self.probability(4); test = self.probability(5)
            _save_component(directory, "C", 0, bindings, valid, test)
            replay_valid, replay_test, _ = _save_component(
                directory, "C", 0, bindings, None, None,
            )
            np.testing.assert_array_equal(replay_valid, valid)
            np.testing.assert_array_equal(replay_test, test)
            changed = dict(bindings, valid_ids_sha256="1" * 64)
            with self.assertRaisesRegex(ValueError, "lineage"):
                _save_component(directory, "C", 0, changed, None, None)

    def test_submission_roundtrip_preserves_id_order_and_known_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = pd.DataFrame({"ID": ["A", "B", "C"], "SUBCLASS": ["x", "x", "x"]})
            probability = np.asarray([[.8, .2], [.1, .9], [.6, .4]], dtype=np.float64)
            path, receipt = _write_submission(
                directory, "A0", sample, probability, ["C0", "C1"], {
                    "selection_inner_macro_f1": .5,
                    "adaptive_full_oof_macro_f1": .49,
                },
            )
            output = pd.read_csv(path)
            self.assertEqual(output["ID"].tolist(), sample["ID"].tolist())
            self.assertEqual(output["SUBCLASS"].tolist(), ["C0", "C1", "C0"])
            self.assertEqual(receipt["rows"], 3)
            self.assertEqual(len(receipt["submission_sha256"]), 64)
            invalid = probability.copy(); invalid[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "invalid"):
                _write_submission(directory, "BAD", sample, invalid, ["C0", "C1"], {})

    def test_candidate_failure_isolated_and_retry_keeps_stage_membership(self):
        declarations = [
            {"candidate_id": "C1", "config_hash": "1" * 64},
            {"candidate_id": "C2", "config_hash": "2" * 64},
        ]
        data = {
            "ids": np.asarray(["A", "B", "C"]),
            "y_label": np.asarray(["C0", "C1", "C2"]),
            "groups": np.asarray(["G0", "G1", "G2"]),
            "class_order": [f"C{i}" for i in range(26)],
        }
        outer = {"fold": 0, "train_indices": [0, 1, 2]}
        probability = np.full((3, 26), 1 / 26, dtype=np.float64)
        def first_pass(rows, *args, **kwargs):
            candidate = rows[0]["candidate_id"]
            if candidate == "C1":
                raise RuntimeError("injected")
            return {"probabilities": {candidate: probability}, "reused_folds": 0}
        def retry(rows, *args, **kwargs):
            candidate = rows[0]["candidate_id"]
            return {"probabilities": {candidate: probability}, "reused_folds": 3}
        with tempfile.TemporaryDirectory() as directory:
            with patch("grade_runner.run_base_bank", side_effect=first_pass):
                complete, failed = _new_stage_probabilities(
                    {}, directory, data, outer, np.asarray([0, 1, 2]), declarations, "a" * 64,
                )
            self.assertEqual(set(complete), {"C2"})
            self.assertEqual(failed, ["C1"])
            with patch("grade_runner.run_base_bank", side_effect=retry):
                complete, failed = _new_stage_probabilities(
                    {}, directory, data, outer, np.asarray([0, 1, 2]), declarations, "a" * 64,
                )
            self.assertEqual(set(complete), {"C1", "C2"})
            self.assertEqual(failed, [])

    def test_ae_wrapper_exposes_actual_head_column_order(self):
        class Autoencoder:
            def cpu(self): return self
            def eval(self): return self
        class Head:
            classes_ = np.asarray(["C2", "C0", "C1"])
        wrapper = AEHeadPredictor(Autoencoder(), Head(), ["C0", "C1", "C2"])
        self.assertEqual(wrapper.classes_.tolist(), ["C2", "C0", "C1"])


if __name__ == "__main__":
    unittest.main()
