from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_grid import MetaCandidate
from promotion import disagreement, promote


def candidate(name, score, labels):
    labels = np.asarray(labels)
    classes = int(labels.max()) + 1
    probability = np.full((len(labels), classes), 0.01)
    probability[np.arange(len(labels)), labels] = 0.98
    probability /= probability.sum(axis=1, keepdims=True)
    return MetaCandidate(name, 0, "A", (), (1.0,), "arithmetic", probability, score, name.ljust(64, "0")[:64])


class PromotionTests(unittest.TestCase):
    def test_fixed_threshold_diversity_order_and_maximum_eight(self):
        base = np.asarray([0, 1] * 100)
        candidates = [candidate("dup-z", 0.9, base), candidate("dup-a", 0.9, base)]
        for index in range(12):
            labels = base.copy()
            start = (index * 17) % len(labels)
            labels[start:start + 8] = 1 - labels[start:start + 8]
            candidates.append(candidate(f"v{index:02d}", 0.89 - index * 0.001, labels))
        candidates.append(candidate("below", 0.549999999, 1 - base))
        retained = promote(candidates)
        self.assertEqual(retained[0].recipe_id, "dup-a")
        self.assertLessEqual(len(retained), 8)
        self.assertNotIn("dup-z", [row.recipe_id for row in retained])
        self.assertNotIn("below", [row.recipe_id for row in retained])
        for left_index, left in enumerate(retained):
            for right in retained[left_index + 1:]:
                self.assertGreaterEqual(disagreement(left, right), 0.03)

    def test_duplicate_predictions_do_not_fill_diversity_slots(self):
        labels = [0, 1] * 30
        retained = promote([candidate("B", 0.7, labels), candidate("A", 0.7, labels)])
        self.assertEqual([row.recipe_id for row in retained], ["A"])


if __name__ == "__main__":
    unittest.main()
