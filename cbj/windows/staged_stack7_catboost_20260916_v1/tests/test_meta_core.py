import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import meta_core
from meta_core import (
    blend,
    declarations,
    features,
    fit_meta,
    predict_meta,
    sample_weights,
)
from tests.catboost_double import FakeCatBoostClassifier, admitted_gpu, patched_backend


def _distinct_probability_fixture():
    return np.asarray(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.7, 0.3], [0.3, 0.7]],
            [[0.6, 0.4], [0.4, 0.6]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.4, 0.6], [0.6, 0.4]],
            [[0.3, 0.7], [0.7, 0.3]],
        ],
        dtype=np.float64,
    )


def _fit_fixture():
    y = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    p = np.empty((7, y.size, 3), dtype=np.float64)
    for model_index in range(7):
        for row_index, label in enumerate(y):
            probability = np.asarray([0.12, 0.12, 0.12], dtype=np.float64)
            probability[label] = 0.76
            shift = 0.005 * model_index
            other = (label + 1) % 3
            probability[label] -= shift
            probability[other] += shift
            p[model_index, row_index] = probability
    return p, y


def _fitted_artifact():
    p, y = _fit_fixture()
    declaration = {
        "representation": "log_probability_confidence",
        "depth": 3,
        "iterations": 200,
        "l2_leaf_reg": 3,
        "weighting": "none",
    }
    with patched_backend():
        return fit_meta(declaration, p, y)


class _NonNormalizedModel(FakeCatBoostClassifier):
    def predict_proba(self, matrix, thread_count=None):
        return np.full((matrix.shape[0], self.classes_.size), 0.2, dtype=np.float64)


class MetaCoreTests(unittest.TestCase):
    def test_declarations_are_the_complete_fixed_cartesian_product(self):
        actual = declarations()
        expected = [
            {
                "representation": "log_probability_confidence",
                "depth": depth,
                "iterations": iterations,
                "l2_leaf_reg": l2_leaf_reg,
                "weighting": weighting,
            }
            for depth in (3, 5, 7)
            for iterations in (200, 600, 1200)
            for l2_leaf_reg in (3, 30)
            for weighting in ("none", "sqrt_inverse", "balanced")
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 54)
        self.assertEqual(
            len({(item["depth"], item["iterations"], item["l2_leaf_reg"], item["weighting"]) for item in actual}),
            54,
        )

    def test_probability_features_use_model_major_then_class_major_order(self):
        p = _distinct_probability_fixture()
        before = p.copy()

        actual = features(p, "probability")

        expected = np.asarray(
            [
                [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7],
                [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(p, before)
        self.assertEqual(actual.dtype, np.dtype(np.float64))

    def test_probability_features_repeat_one_model_vector_seven_times(self):
        p = np.tile(np.asarray([[[0.8, 0.2]]], dtype=np.float64), (7, 1, 1))

        actual = features(p, "probability")

        np.testing.assert_array_equal(actual, np.asarray([[0.8, 0.2] * 7], dtype=np.float64))

    def test_log_probability_confidence_appends_model_major_confidence_tail(self):
        p = np.tile(np.asarray([[[0.8, 0.2]]], dtype=np.float64), (7, 1, 1))
        entropy = -0.8 * np.log(0.8) - 0.2 * np.log(0.2)

        actual = features(p, "log_probability_confidence")

        expected_log = list(np.log([0.8, 0.2])) * 7
        expected_tail = [entropy, 0.8, 0.6] * 7
        np.testing.assert_allclose(
            actual,
            np.asarray([expected_log + expected_tail], dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        )

    def test_log_probability_clips_zero_at_one_e_minus_twelve(self):
        p = np.tile(np.asarray([[[1.0, 0.0]]], dtype=np.float64), (7, 1, 1))

        actual = features(p, "log_probability")

        expected = np.asarray([[0.0, np.log(1e-12)] * 7], dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_features_reject_malformed_probability_tensors(self):
        bad_inputs = [
            np.full((6, 2, 2), 0.5),
            np.full((7, 2, 1), 1.0),
            np.asarray([[[0.6, 0.6]]] * 7),
            np.asarray([[[1.1, -0.1]]] * 7),
            np.asarray([[[np.nan, np.nan]]] * 7),
        ]
        for bad_p in bad_inputs:
            with self.subTest(shape=bad_p.shape), self.assertRaises(ValueError):
                features(bad_p, "probability")

    def test_features_reject_unknown_representation(self):
        p = np.tile(np.asarray([[[0.8, 0.2]]], dtype=np.float64), (7, 1, 1))
        with self.assertRaisesRegex(ValueError, "representation"):
            features(p, "raw_logit")

    def test_sample_weights_match_literal_balanced_and_none_values(self):
        y = np.asarray([0, 0, 0, 1], dtype=np.int64)

        balanced = sample_weights(y, "balanced", classes=2)
        none = sample_weights(y, "none", classes=2)

        np.testing.assert_allclose(balanced, [2 / 3, 2 / 3, 2 / 3, 2], rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(none, np.ones(4, dtype=np.float64))
        self.assertAlmostEqual(float(balanced.mean()), 1.0)

    def test_sample_weights_sqrt_inverse_are_normalized_to_mean_one(self):
        y = np.asarray([0, 0, 0, 1], dtype=np.int64)

        actual = sample_weights(y, "sqrt_inverse", classes=2)

        expected = np.asarray(
            [
                4 / (3 + np.sqrt(3)),
                4 / (3 + np.sqrt(3)),
                4 / (3 + np.sqrt(3)),
                4 / (1 + np.sqrt(3)),
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(actual.mean()), 1.0)

    def test_sample_weights_fail_closed_for_invalid_labels_or_weighting(self):
        cases = [
            (np.asarray([0.0, 1.0]), "none", 2),
            (np.asarray([0, 2]), "none", 2),
            (np.asarray([0, 0]), "none", 2),
            (np.asarray([0, 1]), "inverse", 2),
            (np.asarray([[0, 1]]), "none", 2),
        ]
        for y, weighting, classes in cases:
            with self.subTest(weighting=weighting, shape=y.shape), self.assertRaises((TypeError, ValueError)):
                sample_weights(y, weighting, classes)

    def test_fit_meta_records_effective_settings_and_uses_only_passed_fit_rows_for_scaler(self):
        p, y = _fit_fixture()
        before = p.copy()
        declaration = {
            "representation": "log_probability_confidence",
            "depth": 5,
            "iterations": 600,
            "l2_leaf_reg": 30,
            "weighting": "balanced",
        }

        with patched_backend():
            artifact = fit_meta(declaration, p, y)

        expected_mean = features(p, "log_probability_confidence").mean(axis=0)
        np.testing.assert_allclose(artifact["scaler"].mean_, expected_mean, rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(p, before)
        self.assertEqual(artifact["declaration"], declaration)
        self.assertIsNot(artifact["declaration"], declaration)
        self.assertEqual(artifact["classes"], [0, 1, 2])
        self.assertEqual(artifact["feature_count"], 42)
        self.assertEqual(
            set(artifact),
            {
                "scaler",
                "model",
                "declaration",
                "classes",
                "feature_count",
                "effective_parameters",
                "convergence_warnings",
            },
        )
        effective = artifact["effective_parameters"]
        self.assertEqual(effective["execution"], "GPU_REQUIRED_NO_CPU_FALLBACK")
        self.assertEqual(effective["requested_parameters"], {
            "loss_function": "MultiClass", "iterations": 600, "depth": 5,
            "learning_rate": 0.05, "l2_leaf_reg": 30.0, "task_type": "GPU",
            "devices": "0", "boosting_type": "Plain", "bootstrap_type": "Bernoulli",
            "subsample": 0.8, "border_count": 64, "random_seed": 42,
            "thread_count": 4, "verbose": False, "allow_writing_files": False,
            "use_best_model": False, "gpu_ram_part": 0.65, "classes_count": 3,
        })
        self.assertEqual(effective["tree_count"], 600)
        self.assertEqual(effective["feature_count"], 42)
        self.assertEqual(effective["class_count"], 3)
        self.assertEqual(effective["gpu_admission"], admitted_gpu())
        self.assertEqual(effective["weighting"], "balanced")
        self.assertEqual(effective["scaler"], {"with_mean": True, "with_std": True})
        self.assertIsInstance(artifact["convergence_warnings"], list)
        self.assertTrue(all(isinstance(message, str) for message in artifact["convergence_warnings"]))
        self.assertFalse({"y", "labels", "validation_y", "test_y"}.intersection(artifact))

    def test_predict_meta_returns_normalized_probabilities_in_exact_integer_class_order(self):
        p, _ = _fit_fixture()
        before = p.copy()
        fitted_artifact = _fitted_artifact()

        actual = predict_meta(fitted_artifact, p)

        self.assertEqual(actual.shape, (12, 3))
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0.0).all())
        np.testing.assert_allclose(actual.sum(axis=1), np.ones(12), rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(fitted_artifact["model"].classes_, [0, 1, 2])
        np.testing.assert_array_equal(p, before)

    def test_predict_meta_rejects_missing_or_reordered_class_metadata(self):
        p, _ = _fit_fixture()
        fitted_artifact = _fitted_artifact()
        missing_artifact = copy.deepcopy(fitted_artifact)
        missing_artifact["model"].classes_ = np.asarray([0, 2], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "classes"):
            predict_meta(missing_artifact, p)

        reordered_artifact = copy.deepcopy(fitted_artifact)
        reordered_artifact["classes"] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, "classes"):
            predict_meta(reordered_artifact, p)

    def test_predict_meta_rejects_permuted_saved_model_class_order(self):
        p, _ = _fit_fixture()
        fitted_artifact = _fitted_artifact()
        permuted_artifact = copy.deepcopy(fitted_artifact)
        permuted_artifact["model"].classes_ = np.asarray([2, 0, 1], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "class order"):
            predict_meta(permuted_artifact, p)

    def test_predict_meta_rejects_non_normalized_model_output(self):
        p, _ = _fit_fixture()
        fitted_artifact = _fitted_artifact()
        bad = copy.deepcopy(fitted_artifact["model"])
        bad.__class__ = _NonNormalizedModel
        fitted_artifact["model"] = bad

        with self.assertRaisesRegex(ValueError, "normalized"):
            predict_meta(fitted_artifact, p)

    def test_fit_meta_rejects_invalid_declarations(self):
        p, y = _fit_fixture()
        declarations_to_reject = [
            {"representation": "unknown", "depth": 3, "iterations": 5, "l2_leaf_reg": 3, "weighting": "none"},
            {"representation": "log_probability_confidence", "depth": 0, "iterations": 5, "l2_leaf_reg": 3, "weighting": "none"},
            {"representation": "log_probability_confidence", "depth": 3, "iterations": 0, "l2_leaf_reg": 3, "weighting": "none"},
            {"representation": "log_probability_confidence", "depth": 3, "iterations": 5, "l2_leaf_reg": np.inf, "weighting": "none"},
            {"representation": "log_probability_confidence", "depth": 3, "iterations": 5, "l2_leaf_reg": 3, "weighting": "unknown"},
        ]
        for declaration in declarations_to_reject:
            with self.subTest(declaration=declaration), self.assertRaises((TypeError, ValueError)):
                fit_meta(declaration, p, y)

    def test_fit_meta_rejects_label_count_mismatch_and_missing_class_support(self):
        p, y = _fit_fixture()
        declaration = {"representation": "log_probability_confidence", "depth": 3, "iterations": 5, "l2_leaf_reg": 3, "weighting": "none"}
        with self.assertRaises(ValueError):
            fit_meta(declaration, p, y[:-1])
        with self.assertRaisesRegex(ValueError, "class"):
            fit_meta(declaration, p[:, y != 2], y[y != 2])

    def test_gpu_admission_fails_before_classifier_construction_when_gpu0_is_unavailable(self):
        p, y = _fit_fixture()
        declaration = {"representation": "log_probability_confidence", "depth": 3, "iterations": 5, "l2_leaf_reg": 3, "weighting": "none"}
        FakeCatBoostClassifier.fit_calls = 0
        with mock.patch.object(meta_core, "_load_catboost", return_value=(FakeCatBoostClassifier, lambda: 0)), \
                mock.patch.object(meta_core, "_query_gpu0", return_value=admitted_gpu()), \
                self.assertRaisesRegex(RuntimeError, "GPU0"):
            fit_meta(declaration, p, y)
        self.assertEqual(FakeCatBoostClassifier.fit_calls, 0)

    def test_predict_meta_rejects_saved_gpu_parameter_or_feature_metadata_tamper(self):
        p, _ = _fit_fixture()
        artifact = _fitted_artifact()
        artifact["model"].parameters["task_type"] = "CPU"
        with self.assertRaisesRegex(ValueError, "task_type"):
            predict_meta(artifact, p)

        artifact = _fitted_artifact()
        artifact["model"].n_features_in_ = 0
        artifact["model"].feature_names_.pop()
        with self.assertRaisesRegex(ValueError, "feature"):
            predict_meta(artifact, p)

    def test_predict_meta_rejects_fixed_effective_parameter_tamper(self):
        p, _ = _fit_fixture()
        tampered_values = {
            "learning_rate": 0.07,
            "subsample": 0.9,
            "devices": "1",
            "gpu_ram_part": 0.5,
            "use_best_model": True,
            "classes_count": 4,
        }
        for key, value in tampered_values.items():
            artifact = _fitted_artifact()
            artifact["model"].effective_parameters[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                predict_meta(artifact, p)

    def test_blend_arithmetic_literal_and_guard_restore_baseline(self):
        baseline = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
        meta = np.asarray([[0.2, 0.8], [0.1, 0.9]], dtype=np.float64)
        guard = np.asarray([False, True])
        baseline_before = baseline.copy()
        meta_before = meta.copy()

        actual = blend(baseline, meta, alpha=0.25, pooling="arithmetic", guard=guard)

        np.testing.assert_allclose(actual[0], [0.65, 0.35], rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(actual[1], baseline[1])
        np.testing.assert_array_equal(baseline, baseline_before)
        np.testing.assert_array_equal(meta, meta_before)

    def test_blend_has_canonical_exact_endpoints_for_both_poolings(self):
        baseline = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
        meta = np.asarray([[0.2, 0.8], [0.1, 0.9]], dtype=np.float64)
        guard = np.asarray([False, True])

        alpha_zero_arithmetic = blend(baseline, meta, 0.0, "arithmetic", guard)
        alpha_zero_geometric = blend(baseline, meta, 0.0, "geometric", guard)
        alpha_one_arithmetic = blend(baseline, meta, 1.0, "arithmetic", guard)
        alpha_one_geometric = blend(baseline, meta, 1.0, "geometric", guard)

        np.testing.assert_array_equal(alpha_zero_arithmetic, baseline)
        np.testing.assert_array_equal(alpha_zero_geometric, baseline)
        np.testing.assert_array_equal(alpha_one_arithmetic, alpha_one_geometric)
        np.testing.assert_array_equal(alpha_one_arithmetic, [[0.2, 0.8], [0.7, 0.3]])

    def test_interior_arithmetic_blend_normalizes_accepted_rounding_drift(self):
        baseline = np.asarray([[0.8 + 5e-9, 0.2]], dtype=np.float64)
        meta = np.asarray([[0.2, 0.8]], dtype=np.float64)

        actual = blend(baseline, meta, 0.25, "arithmetic", np.asarray([False]))

        np.testing.assert_allclose(actual.sum(axis=1), [1.0], rtol=0.0, atol=1e-15)

    def test_blend_fails_closed_for_invalid_inputs(self):
        cases = [
            ([[0.8, 0.2]], [[0.2, 0.8]], -0.1, "arithmetic", [False]),
            ([[0.8, 0.2]], [[0.2, 0.8]], 1.1, "arithmetic", [False]),
            ([[0.8, 0.2]], [[0.2, 0.8]], np.nan, "arithmetic", [False]),
            ([[0.8, 0.2]], [[0.2, 0.8]], 0.5, "harmonic", [False]),
            ([[0.8, 0.3]], [[0.2, 0.8]], 0.5, "arithmetic", [False]),
            ([[0.8, 0.2]], [[0.2, np.nan]], 0.5, "arithmetic", [False]),
            ([[0.8, 0.2]], [[0.2, 0.8], [0.5, 0.5]], 0.5, "arithmetic", [False]),
            ([[0.8, 0.2]], [[0.2, 0.8]], 0.5, "arithmetic", [False, True]),
            ([[0.8, 0.2]], [[0.2, 0.8]], 0.5, "arithmetic", [0]),
        ]
        for baseline, meta, alpha, pooling, guard in cases:
            with self.subTest(alpha=alpha, pooling=pooling, guard=guard), self.assertRaises(
                (TypeError, ValueError)
            ):
                blend(
                    np.asarray(baseline),
                    np.asarray(meta),
                    alpha,
                    pooling,
                    np.asarray(guard),
                )


if __name__ == "__main__":
    unittest.main()
