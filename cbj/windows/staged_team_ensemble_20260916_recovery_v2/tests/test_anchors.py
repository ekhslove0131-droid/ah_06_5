from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchor_pipeline import (
    anchor_model_recipes,
    build_anchor_oof,
    build_anchor_predictions,
    prepare_anchor_features,
    replay_historical_anchors,
)
from support_guard import geometric_pool, guarded_probability, support_guard_mask
from tests.test_feature_transforms import fixture


HISTORICAL_CANDIDATES = (
    ROOT.parent / "goal055_sol/results/goal055_nested_macro_bias_seed43_v2",
    ROOT.parent / "runs/goal055_nested_macro_bias_seed43_v2",
)
HISTORICAL = next((path for path in HISTORICAL_CANDIDATES if path.is_dir()), HISTORICAL_CANDIDATES[0])


class AnchorTests(unittest.TestCase):
    def test_geometric_pool_and_guard_are_exact(self):
        h1 = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        c10 = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float64)
        pooled = geometric_pool(h1, c10)
        expected = np.sqrt(h1 * c10); expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(pooled, expected, atol=1e-15)
        guarded = guarded_probability(pooled, h1, np.asarray([True, False]))
        np.testing.assert_array_equal(guarded[0], h1[0])
        np.testing.assert_array_equal(guarded[1], pooled[1])

    def test_support_mask_uses_only_symbolic_zero_rows(self):
        train = sparse.csr_matrix([[1, 0], [0, 1]])
        valid = sparse.csr_matrix([[0, 0], [1, 0]])
        mask, receipt = support_guard_mask(train, valid)
        np.testing.assert_array_equal(mask, [True, False])
        self.assertFalse(receipt["labels_or_ids_accepted_by_rule"])
        mask_with_train_zero, _ = support_guard_mask(
            sparse.csr_matrix([[0, 0], [1, 0]]), valid,
        )
        np.testing.assert_array_equal(mask_with_train_zero, [False, False])

    def test_two_anchor_branches_share_inputs_but_only_a2_applies_inner_bias(self):
        rng = np.random.default_rng(7)
        h1 = rng.random((30, 3)); h1 /= h1.sum(axis=1, keepdims=True)
        c10 = rng.random((30, 3)); c10 /= c10.sum(axis=1, keepdims=True)
        inner = rng.random((60, 3)); inner /= inner.sum(axis=1, keepdims=True)
        inner_y = np.asarray([index % 3 for index in range(60)])
        mask = np.zeros(30, dtype=bool); mask[:2] = True
        result = build_anchor_predictions(h1, c10, mask, inner, inner_y)
        self.assertEqual(set(result), {"A1_GEOMETRIC_GUARD", "A2_C10_BIAS_GEOMETRIC_GUARD", "bias_receipt"})
        self.assertFalse(np.array_equal(result["A1_GEOMETRIC_GUARD"], result["A2_C10_BIAS_GEOMETRIC_GUARD"]))
        np.testing.assert_array_equal(result["A1_GEOMETRIC_GUARD"][:2], h1[:2])
        np.testing.assert_array_equal(result["A2_C10_BIAS_GEOMETRIC_GUARD"][:2], h1[:2])
        self.assertNotIn("outer_valid_y", inspect.signature(build_anchor_predictions).parameters)

    def test_anchor_model_recipes_are_the_frozen_h1_and_c10_models(self):
        recipes = anchor_model_recipes()
        h1 = recipes["H1"]
        c10 = recipes["C10"]
        self.assertEqual(h1["kind"], "xgboost")
        self.assertEqual(h1["device_kind"], "GPU_REQUIRED")
        self.assertEqual(h1["weighting"], "sqrt_inverse")
        self.assertEqual(h1["params"]["n_estimators"], 200)
        self.assertEqual(h1["params"]["max_depth"], 3)
        self.assertEqual(h1["params"]["learning_rate"], 0.1)
        self.assertEqual(c10["kind"], "logistic")
        self.assertEqual(c10["device_kind"], "CPU_INTENTIONAL")
        self.assertEqual(c10["weighting"], "sqrt_inverse")
        self.assertEqual(c10["params"]["C"], 10.0)
        self.assertEqual(c10["params"]["max_iter"], 2000)

    def test_a2_inner_validation_labels_are_not_an_api_input(self):
        rng = np.random.default_rng(19)
        h1 = rng.random((12, 3)); h1 /= h1.sum(axis=1, keepdims=True)
        c10 = rng.random((12, 3)); c10 /= c10.sum(axis=1, keepdims=True)
        subinner = rng.random((60, 3)); subinner /= subinner.sum(axis=1, keepdims=True)
        subinner_y = np.asarray([index % 3 for index in range(60)])
        first = build_anchor_oof(
            h1, c10, np.zeros(12, dtype=bool), subinner, subinner_y,
            branch="A2_C10_BIAS_GEOMETRIC_GUARD",
        )
        # These are the held-out labels an unsafe API might accidentally use.
        heldout_y = np.asarray([index % 3 for index in range(12)])
        heldout_y[:] = heldout_y[::-1]
        second = build_anchor_oof(
            h1, c10, np.zeros(12, dtype=bool), subinner, subinner_y,
            branch="A2_C10_BIAS_GEOMETRIC_GUARD",
        )
        np.testing.assert_array_equal(first["probability"], second["probability"])
        self.assertNotIn("valid_y", inspect.signature(build_anchor_oof).parameters)
        self.assertFalse(first["receipt"]["outer_or_same_validation_labels_accepted"])

    def test_historical_anchor_oof_scores_and_hashes_replay(self):
        replay = replay_historical_anchors(HISTORICAL)
        self.assertAlmostEqual(replay["A1_GEOMETRIC_GUARD"]["macro_f1"], 0.5127255062867847)
        self.assertAlmostEqual(replay["A2_C10_BIAS_GEOMETRIC_GUARD"]["macro_f1"], 0.5204059148667912)
        self.assertEqual(
            replay["A1_GEOMETRIC_GUARD"]["sha256"],
            "3b668367ca6d48f9c05f2a507458b6e0d530322722c99422980aba353722084a",
        )
        self.assertEqual(
            replay["A2_C10_BIAS_GEOMETRIC_GUARD"]["sha256"],
            "f4ede5b52937a5c8fc5902413da2e6a15b0add0fb22dc20a6f80a639cd90e517",
        )
        self.assertEqual(replay["guarded_rows"], 94)
        self.assertEqual(
            replay["A1_GEOMETRIC_GUARD"]["argmax_sha256"],
            "1f8f42adfac600311cf7f661e99c30c9cd2029eee2affcc46759c8b7edaacf96",
        )
        self.assertEqual(
            replay["A2_C10_BIAS_GEOMETRIC_GUARD"]["argmax_sha256"],
            "26c51cae601da997e52329d68637aa626fc1fadd9e0ced79fc6deff9d77c528f",
        )

    def test_anchor_feature_preparation_is_fold_train_fitted(self):
        train, valid, context = fixture()
        first = prepare_anchor_features(train, valid, context)
        changed = valid.copy(); changed.iloc[:, :] = "VALID_ONLY999V"
        second = prepare_anchor_features(train, changed, context)
        self.assertEqual(first["H1"].state["fitted"], second["H1"].state["fitted"])
        self.assertEqual(first["C10"].state["fitted"], second["C10"].state["fitted"])
        self.assertEqual(first["H1"].state["selected_k"], min(1200, first["H1"].state["prefilter_pool"]))
        self.assertEqual(first["C10"].state["C"], 10.0)
        self.assertNotIn("valid_y", inspect.signature(prepare_anchor_features).parameters)


if __name__ == "__main__":
    unittest.main()
