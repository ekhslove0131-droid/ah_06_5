from __future__ import annotations

import os
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_transforms import FeaturePair
from model_adapters import (
    GPURequiredError, effective_recipe, fit_model, load_artifact,
    predict_checked, save_artifact,
)
from neural_candidates import PreparedNeuralCandidate
from tests.test_feature_transforms import declaration
from tests.test_feature_transforms import fixture
from feature_transforms import fit_transform_candidate
from workflow import _fit_prepared
from tests.test_old_candidates import old_declaration


NATIVE = os.environ.get("CBJ_GPU_NATIVE_TEST") == "1"


def dense_pair():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(30, 20)).astype(np.float32)
    valid = rng.normal(size=(9, 20)).astype(np.float32)
    return FeaturePair(
        train=train, valid=valid, feature_names=[f"f{i}" for i in range(20)],
        state={}, sample_weight_multiplier=np.ones(30, np.float32),
        train_sha256="1" * 64, valid_sha256="2" * 64,
        deferred_numeric_train=np.abs(rng.normal(size=(30, 41))).astype(np.float32),
        deferred_numeric_valid=np.abs(rng.normal(size=(9, 41))).astype(np.float32),
    )


@unittest.skipUnless(NATIVE, "requires explicit Windows CUDA native test mode")
class NativeGPUModelTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["C0", "C1", "C2"]
        self.y = np.asarray([self.classes[index % 3] for index in range(30)])

    def assert_gpu_fit(self, declaration_value, features, valid_payload):
        artifact = fit_model(
            effective_recipe(declaration_value), features, self.y, None, self.classes,
            capabilities={"cuda": True}, device="cuda:0",
        )
        self.assertEqual(artifact.device, "cuda:0")
        probabilities = predict_checked(artifact, valid_payload, self.classes)
        self.assertEqual(probabilities.shape, (9, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            receipt = save_artifact(artifact, path)
            restored = load_artifact(path, receipt)
            replay = predict_checked(restored, valid_payload, self.classes)
            np.testing.assert_allclose(probabilities, replay, atol=1e-6)
            np.testing.assert_array_equal(probabilities.argmax(1), replay.argmax(1))

    def test_xgboost_is_native_gpu(self):
        pair = dense_pair()
        self.assert_gpu_fit(
            declaration("T1", storage="dense_explicit_zero", burden="none", class_weight="sqrt_inverse"),
            pair, pair.valid,
        )

    def test_lightgbm_without_gpu_build_fails_closed(self):
        pair = dense_pair()
        with self.assertRaisesRegex(GPURequiredError, "GPU Tree Learner"):
            fit_model(
                effective_recipe(declaration(
                    "T4", input="our_B0_B1_support3_TFIDF", compressor="SVD64", head="LightGBM_fixed",
                )), pair, self.y, None, self.classes,
                capabilities={"cuda": True}, device="cuda:0",
            )

    def test_ae_fm_and_token_models_train_and_predict_on_cuda(self):
        pair = dense_pair()
        self.assert_gpu_fit(
            declaration("T4", input="kim_ordinal_chi2top500", compressor="AE64", head="logreg_C1"),
            pair, {"ae_input": pair.valid, "numeric41": pair.deferred_numeric_valid},
        )
        rng = np.random.default_rng(7)
        fm_features = PreparedNeuralCandidate(
            train={
                "symbolic": rng.normal(size=(30, 12)).astype(np.float32),
                "missing": rng.integers(0, 2, size=(30, 4)).astype(np.float32),
                "numeric41": rng.normal(size=(30, 41)).astype(np.float32),
            },
            valid={
                "symbolic": rng.normal(size=(9, 12)).astype(np.float32),
                "missing": rng.integers(0, 2, size=(9, 4)).astype(np.float32),
                "numeric41": rng.normal(size=(9, 41)).astype(np.float32),
            },
            state={"fitted": {}, "architecture": {}},
        )
        self.assert_gpu_fit(
            old_declaration("G5", view="gene_binary", interaction_rank=8, weight_decay=0.001),
            fm_features, fm_features.valid,
        )
        token_rows_train = [[{"gene": 1, "kind": 1, "aa_from": 0, "aa_to": 2, "position": 0.1, "unknown_position": 0.0}]] * 30
        token_rows_valid = [[{"gene": 1, "kind": 1, "aa_from": 0, "aa_to": 2, "position": 0.2, "unknown_position": 0.0}]] * 9
        token_features = PreparedNeuralCandidate(
            train={"tokens": token_rows_train, "numeric41": rng.normal(size=(30, 41)).astype(np.float32)},
            valid={"tokens": token_rows_valid, "numeric41": rng.normal(size=(9, 41)).astype(np.float32)},
            state={
                "fitted": {
                    "gene_vocabulary": ["G"], "kind_vocabulary": ["missense"],
                    "aa_from_vocabulary": ["A"], "aa_to_vocabulary": ["V", "W"],
                },
                "architecture": {},
            },
        )
        self.assert_gpu_fit(
            old_declaration("G6", token_embedding_dim=32, pool="mean_max", dropout=0.3),
            token_features, token_features.valid,
        )

    def test_real_t4_workflow_transform_fit_and_predict(self):
        """Exercise the exact T4 transform -> AE64 -> logistic prediction path."""
        train, valid, context = fixture(rows=90, genes=70)
        declaration_value = declaration(
            "T4", input="kim_ordinal_chi2top500", compressor="AE64", head="logreg_C1",
        )
        features = fit_transform_candidate(declaration_value, train, valid, context)
        recipe = effective_recipe(declaration_value)
        probability = _fit_prepared(recipe, features, context)
        self.assertEqual(probability.shape, (len(valid), len(context.class_order)))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
