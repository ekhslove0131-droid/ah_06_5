from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raw_views import (
    exact_cell_token_rows,
    literal_gene_binary,
    mutation_severity,
    recurrent_tokens,
)


class RawViewTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "G1": ["WT", "A1V", None, "A1V"],
            "G2": ["", "A1A; B2*", "WT", "A1V"],
        })

    def test_aeyoung_literal_binary_preserves_exact_wt_and_missing_semantics(self):
        observed = literal_gene_binary(self.frame(), ["G1", "G2"])
        np.testing.assert_array_equal(observed, np.asarray([
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 1],
        ], dtype=np.float32))

    def test_kim_exact_hash_tokens_keep_original_full_cell_string(self):
        rows = exact_cell_token_rows(self.frame(), ["G1", "G2"])
        self.assertEqual(rows[0], ["G2="])
        self.assertEqual(rows[1], ["G1=A1V", "G2=A1A; B2*"])
        self.assertEqual(rows[2], [])

    def test_legacy_severity_and_hotspot_support_are_train_only(self):
        train = self.frame().iloc[[1, 3, 3]].reset_index(drop=True)
        hot = recurrent_tokens(train, ["G1", "G2"], min_count=2)
        self.assertEqual(hot, {"G1": {"A1V"}, "G2": {"A1V"}})
        self.assertEqual(mutation_severity(None), -1.0)
        self.assertEqual(mutation_severity("WT"), 0.0)
        self.assertEqual(mutation_severity("A1A"), 1.0)
        self.assertEqual(mutation_severity("A1V"), 2.0)
        self.assertEqual(mutation_severity("A1fs"), 3.0)
        self.assertEqual(mutation_severity("A1V", {"A1V"}), 4.0)
        self.assertEqual(mutation_severity("A1V B2C"), 2.15)


if __name__ == "__main__":
    unittest.main()
