from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_transforms import FeaturePair
from model_adapters import (
    GPURequiredError,
    ModelArtifact,
    effective_recipe,
    fit_model,
    load_artifact,
    predict_checked,
    save_artifact,
)
from tests.test_feature_transforms import declaration
from tests.test_old_candidates import old_declaration
from workflow import _fit_prepared


def pair(rows=60, cols=8, classes=3):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(rows, cols)).astype(np.float32)
    y = np.asarray([f"C{i % classes}" for i in range(rows)])
    feature = FeaturePair(
        train=X[:48], valid=X[48:], feature_names=[f"f{i}" for i in range(cols)],
        state={}, sample_weight_multiplier=np.ones(48, np.float32),
        train_sha256="a" * 64, valid_sha256="b" * 64,
    )
    return feature, y[:48], [f"C{i}" for i in range(classes)]


class ModelAdapterTests(unittest.TestCase):
    def test_workflow_passes_ae_input_and_deferred_numeric_to_ae_predictor(self):
        classes = ["C0", "C1", "C2"]
        feature, y, _ = pair()
        feature.deferred_numeric_train = np.zeros((48, 41), dtype=np.float32)
        feature.deferred_numeric_valid = np.ones((12, 41), dtype=np.float32)
        recipe = effective_recipe(declaration(
            "T4", input="kim_ordinal_chi2top500", compressor="AE64", head="logreg_C1",
        ))

        class PayloadCheckingModel:
            def predict_proba(self, payload):
                np.testing.assert_array_equal(payload["ae_input"], feature.valid)
                np.testing.assert_array_equal(payload["numeric41"], feature.deferred_numeric_valid)
                return np.full((len(feature.valid), len(classes)), 1 / len(classes), dtype=np.float64)

        artifact = ModelArtifact(
            PayloadCheckingModel(), recipe, classes, [], "a" * 64, "cuda:0",
        )
        context = SimpleNamespace(train_y=y, class_order=classes)
        with patch("workflow.fit_model", return_value=artifact):
            probability = _fit_prepared(recipe, feature, context)
        self.assertEqual(probability.shape, (12, 3))

    def test_effective_parameters_are_fully_pinned(self):
        recipes = [
            effective_recipe(declaration("T1", storage="dense_explicit_zero", burden="none", class_weight="sqrt_inverse")),
            effective_recipe(declaration("T2", view="presence_burden", min_samples_leaf=5, max_features=0.1)),
            effective_recipe(declaration("T4", input="kim_ordinal_chi2top500", compressor="AE64", head="LightGBM_fixed")),
            effective_recipe(old_declaration("G1", position_bin_width=8, min_group_support=2, C=10)),
            effective_recipe(old_declaration("G5", view="gene_binary", interaction_rank=32, weight_decay=0.01)),
            effective_recipe(old_declaration("G6", token_embedding_dim=64, pool="mean_max", dropout=0.5)),
        ]
        self.assertEqual(recipes[0]["params"]["n_estimators"], 200)
        self.assertEqual(recipes[0]["params"]["max_depth"], 3)
        self.assertEqual(recipes[1]["params"]["n_estimators"], 500)
        self.assertEqual(recipes[1]["params"]["criterion"], "entropy")
        self.assertEqual(recipes[2]["transformer"]["epochs"], 80)
        self.assertEqual(recipes[2]["params"]["num_leaves"], 15)
        self.assertEqual(recipes[3]["params"]["C"], 10.0)
        self.assertEqual(recipes[4]["params"]["interaction_rank"], 32)
        self.assertEqual(recipes[5]["params"]["token_embedding_dim"], 64)

    def test_gpu_recipes_fail_before_fit_without_cuda(self):
        feature, y, classes = pair()
        for declaration_value in (
            declaration("T1", storage="dense_explicit_zero", burden="none", class_weight="none"),
            declaration("T4", input="kim_ordinal_chi2top500", compressor="AE64", head="LightGBM_fixed"),
            old_declaration("G5", view="gene_binary", interaction_rank=8, weight_decay=0.001),
        ):
            with self.assertRaisesRegex(GPURequiredError, "CUDA"):
                fit_model(effective_recipe(declaration_value), feature, y, None, classes, capabilities={"cuda": False})

    def test_cpu_logistic_and_extratrees_fit_predict_and_reload(self):
        feature, y, classes = pair()
        for declaration_value in (
            old_declaration("G1", position_bin_width=8, min_group_support=2, C=1),
            declaration("T2", view="presence_burden", min_samples_leaf=2, max_features="sqrt"),
        ):
            recipe = effective_recipe(declaration_value)
            artifact = fit_model(recipe, feature, y, None, classes, capabilities={"cuda": False})
            before = predict_checked(artifact, feature.valid, classes)
            self.assertEqual(before.shape, (12, 3))
            np.testing.assert_allclose(before.sum(axis=1), 1.0, atol=1e-7)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.joblib"
                receipt = save_artifact(artifact, path)
                restored = load_artifact(path, receipt)
                after = predict_checked(restored, feature.valid, classes)
                np.testing.assert_allclose(before, after, atol=1e-12)
                np.testing.assert_array_equal(before.argmax(1), after.argmax(1))

    def test_g3_ovr_uses_class_specific_reweights(self):
        feature, y, classes = pair(cols=5)
        feature.symbolic_width = 3
        feature.class_reweights = np.asarray([
            [1, 1, 1], [1, -1, 0.5], [-0.5, 1, 2],
        ], dtype=np.float32)
        recipe = effective_recipe(old_declaration("G3", view="B0", reweight="nb_alpha1", C=0.3))
        artifact = fit_model(recipe, feature, y, None, classes, capabilities={"cuda": False})
        probabilities = predict_checked(artifact, feature.valid, classes)
        self.assertEqual(probabilities.shape, (12, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
