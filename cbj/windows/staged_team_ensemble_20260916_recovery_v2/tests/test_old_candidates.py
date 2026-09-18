from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from old_candidates import fit_transform_old_candidate
from tests.test_feature_transforms import fixture


def old_declaration(family, **parameters):
    return {
        "candidate_id": f"OLD-{family}-TEST",
        "config_hash": "b" * 64,
        "source_kind": "OLD",
        "family": family,
        "parameters": parameters,
        "fixed": {},
    }


class OldCandidateTests(unittest.TestCase):
    def test_g1_position_bins_use_safe_start_and_train_fitted_vocabulary(self):
        train, valid, context = fixture()
        pair = fit_transform_old_candidate(old_declaration(
            "G1", position_bin_width=8, min_group_support=2, C=1,
        ), train, valid, context)
        self.assertIn("position_bin|G008|missense|1", pair.feature_names)
        self.assertTrue(all("unresolved" not in name for name in pair.feature_names if name.startswith("position_bin|")))
        changed = valid.copy(); changed.iloc[:, :] = "A999V"
        replay = fit_transform_old_candidate(old_declaration(
            "G1", position_bin_width=8, min_group_support=2, C=1,
        ), train, changed, context)
        self.assertEqual(pair.state["fitted"], replay.state["fitted"])

    def test_g1_group_support_is_counted_after_position_binning(self):
        train, valid, context = fixture()
        train.loc[:, "G000"] = "WT"
        train.loc[0, "G000"] = "A1V"
        train.loc[1, "G000"] = "A2V"
        pair = fit_transform_old_candidate(old_declaration(
            "G1", position_bin_width=8, min_group_support=2, C=1,
        ), train, valid, context)
        self.assertIn("position_bin|G000|missense|0", pair.feature_names)

    def test_g2_uses_independent_blocks_and_declared_fine_weight(self):
        train, valid, context = fixture()
        pair = fit_transform_old_candidate(old_declaration(
            "G2", fine_min_group_support=2, fine_block_weight=0.25, C=3,
        ), train, valid, context)
        fitted = pair.state["fitted"]
        self.assertEqual(fitted["fine_weight"], 0.25)
        self.assertEqual(fitted["coarse_tfidf"]["norm"], "l2")
        self.assertEqual(fitted["fine_tfidf"]["norm"], "l2")
        self.assertEqual(pair.train.shape[1], len(pair.feature_names))
        self.assertEqual(pair.feature_names[-1], "numeric41::40")

    def test_g3_nb_ratios_are_class_specific_and_plain_control_is_one(self):
        train, valid, context = fixture()
        plain = fit_transform_old_candidate(old_declaration(
            "G3", view="B0", reweight="none", C=0.3,
        ), train, valid, context)
        nb = fit_transform_old_candidate(old_declaration(
            "G3", view="B0_B1_support3", reweight="nb_alpha1", C=3,
        ), train, valid, context)
        self.assertEqual(plain.class_reweights.shape, (3, plain.symbolic_width))
        np.testing.assert_array_equal(plain.class_reweights, np.ones_like(plain.class_reweights))
        self.assertEqual(nb.class_reweights.shape, (3, nb.symbolic_width))
        self.assertTrue(np.isfinite(nb.class_reweights).all())
        self.assertFalse(np.allclose(nb.class_reweights[0], nb.class_reweights[1]))

    def test_g4_svd_and_nmf_append_numeric41_at_exact_rank(self):
        train, valid, context = fixture()
        svd = fit_transform_old_candidate(old_declaration(
            "G4", decomposition="TruncatedSVD", components=32, C=0.3,
        ), train, valid, context)
        nmf = fit_transform_old_candidate(old_declaration(
            "G4", decomposition="NMF", components=32, C=3,
        ), train, valid, context)
        self.assertEqual(svd.train.shape[1], 32 + 41)
        self.assertEqual(nmf.train.shape[1], 32 + 41)
        self.assertTrue(np.isfinite(svd.train).all())
        self.assertTrue(np.isfinite(nmf.train).all())


if __name__ == "__main__":
    unittest.main()
