from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural_candidates import prepare_neural_candidate
from tests.test_feature_transforms import fixture
from tests.test_old_candidates import old_declaration


class NeuralCandidateTests(unittest.TestCase):
    def test_g5_binary_rows_are_l2_normalized_and_missing_flags_preserved(self):
        train, valid, context = fixture()
        prepared = prepare_neural_candidate(old_declaration(
            "G5", view="gene_binary", interaction_rank=16, weight_decay=0.01,
        ), train, valid, context)
        norms = np.linalg.norm(prepared.train["symbolic"], axis=1)
        self.assertTrue(np.all((np.isclose(norms, 1.0)) | (np.isclose(norms, 0.0))))
        self.assertEqual(prepared.train["missing"].shape, (len(train), 70))
        self.assertEqual(prepared.state["architecture"]["interaction_rank"], 16)
        self.assertEqual(prepared.state["architecture"]["epochs"], 80)

    def test_g6_vocabulary_uses_train_group_support_and_preserves_repeated_tokens(self):
        train, valid, context = fixture()
        train.loc[0, "G000"] = "A1V A1V"
        train.loc[1, "G000"] = "A1V"
        train.loc[:, "G069"] = "WT"
        prepared = prepare_neural_candidate(old_declaration(
            "G6", token_embedding_dim=32, pool="mean_sqrt_normalized_sum", dropout=0.3,
        ), train, valid, context)
        first_tokens = prepared.train["tokens"][0]
        self.assertGreaterEqual(len(first_tokens), 2)
        self.assertEqual(first_tokens[0], first_tokens[1])
        self.assertIn("G000", prepared.state["fitted"]["gene_vocabulary"])
        changed = valid.copy(); changed.loc[0, "G069"] = "VALIDONLY999V"
        replay = prepare_neural_candidate(old_declaration(
            "G6", token_embedding_dim=32, pool="mean_sqrt_normalized_sum", dropout=0.3,
        ), train, changed, context)
        self.assertEqual(prepared.state["fitted"], replay.state["fitted"])
        self.assertTrue(any(token["gene"] == 0 for token in replay.valid["tokens"][0]))

    def test_g5_and_g6_contracts_require_gpu_training(self):
        train, valid, context = fixture()
        for declaration in (
            old_declaration("G5", view="gene_type_binary", interaction_rank=8, weight_decay=0.001),
            old_declaration("G6", token_embedding_dim=64, pool="mean_max", dropout=0.5),
        ):
            prepared = prepare_neural_candidate(declaration, train, valid, context)
            self.assertEqual(prepared.state["device_kind"], "GPU_REQUIRED")
            self.assertFalse(prepared.state["token_truncation"])


if __name__ == "__main__":
    unittest.main()
