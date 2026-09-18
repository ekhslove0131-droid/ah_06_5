from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cache_store import CacheIntegrityError, ImmutableProbabilityCache, make_cache_key


def bindings(**changes):
    base = {
        "input_sha256": "1" * 64,
        "parser_sha256": "2" * 64,
        "source_sha256": "3" * 64,
        "declaration_sha256": "4" * 64,
        "fit_ids_sha256": "5" * 64,
        "fit_y_sha256": "6" * 64,
        "fit_groups_sha256": "7" * 64,
        "partition_sha256": "8" * 64,
        "feature_order_sha256": "9" * 64,
        "parameters_sha256": "a" * 64,
    }
    base.update(changes)
    return base


class CacheStoreTests(unittest.TestCase):
    def test_supervised_key_requires_every_lineage_hash_and_changes_on_collision(self):
        key = make_cache_key(bindings(), supervised=True)
        self.assertEqual(len(key), 64)
        self.assertNotEqual(key, make_cache_key(bindings(source_sha256="b" * 64), supervised=True))
        missing = bindings(); del missing["fit_y_sha256"]
        with self.assertRaisesRegex(ValueError, "fit_y_sha256"):
            make_cache_key(missing, supervised=True)

    def test_label_free_key_omits_labels_but_retains_ids_and_groups(self):
        raw = bindings(); raw.pop("fit_y_sha256")
        key = make_cache_key(raw, supervised=False)
        changed = dict(raw, fit_groups_sha256="b" * 64)
        self.assertNotEqual(key, make_cache_key(changed, supervised=False))

    def test_completed_artifact_is_adopted_after_crash_before_ledger(self):
        probability = np.asarray([[0.7, 0.3], [0.2, 0.8]])
        with tempfile.TemporaryDirectory() as directory:
            cache = ImmutableProbabilityCache(directory)
            key = make_cache_key(bindings(), supervised=True)
            cache.write(key, probability, bindings(), stop_before_ledger=True)
            self.assertFalse(cache.ledger_path(key).exists())
            loaded, state = cache.load_or_adopt(key, bindings())
            np.testing.assert_array_equal(loaded, probability)
            self.assertEqual(state, "ADOPTED")
            self.assertTrue(cache.ledger_path(key).exists())
            loaded, state = cache.load_or_adopt(key, bindings())
            self.assertEqual(state, "REUSED")

    def test_tampered_probability_is_rejected(self):
        probability = np.asarray([[0.7, 0.3], [0.2, 0.8]])
        with tempfile.TemporaryDirectory() as directory:
            cache = ImmutableProbabilityCache(directory)
            key = make_cache_key(bindings(), supervised=True)
            cache.write(key, probability, bindings())
            np.save(cache.artifact_path(key) / "probability.npy", probability[::-1])
            with self.assertRaisesRegex(CacheIntegrityError, "hash"):
                cache.load_or_adopt(key, bindings())


if __name__ == "__main__":
    unittest.main()
