from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from numeric_views import build_numeric41
from tests.test_feature_transforms import fixture


class NumericViewTests(unittest.TestCase):
    def test_numeric41_is_exact_width_nonnegative_and_fit_boundary_safe(self):
        train, valid, context = fixture()
        first = build_numeric41(train, valid, context.gene_cols, context.train_groups)
        changed = valid.copy(); changed.iloc[:, :] = "VALID_ONLY999V"
        second = build_numeric41(train, changed, context.gene_cols, context.train_groups)
        self.assertEqual(first["train"].shape, (len(train), 41))
        self.assertEqual(first["valid"].shape, (len(valid), 41))
        self.assertEqual(len(first["names"]), 41)
        self.assertTrue(np.isfinite(first["train"]).all())
        self.assertTrue((first["train"] >= 0).all())
        np.testing.assert_array_equal(first["train"], second["train"])
        self.assertEqual(first["fitted"], second["fitted"])
        self.assertFalse(np.array_equal(first["valid"], second["valid"]))
        self.assertFalse(first["receipt"]["labels_accepted"])


if __name__ == "__main__":
    unittest.main()
