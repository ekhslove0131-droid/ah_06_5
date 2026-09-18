from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from combo_core import active_leaf_ids, evaluate, pool, simplex, translate_v2_selection


class ComboCoreTests(unittest.TestCase):
    def test_simplex_six_has_exact_126_declared_compositions(self):
        values = simplex(slots=6)
        self.assertEqual(len(values), 126)
        self.assertEqual(len(set(values)), 126)
        self.assertTrue(all(abs(sum(row) - 1) < 1e-12 for row in values))

    def test_zero_weight_nested_sources_are_not_loaded(self):
        nodes = {
            "a": {"kind": "leaf", "leaf_id": "A"},
            "missing": {"kind": "leaf", "leaf_id": "MISSING"},
            "inner": {"kind": "mix", "sources": ["a", "missing"], "weights": [1, 0], "pooling": "geometric"},
            "root": {"kind": "mix", "sources": ["inner", "missing"], "weights": [1, 0], "pooling": "arithmetic"},
        }
        p = np.asarray([[.8, .2], [.1, .9]])
        leaves = {"A": p, "BASE": p, "GUARD": np.asarray([False, False])}
        recipe = {"root": "root", "guard_baseline_leaf": "BASE", "guard_mask_leaf": "GUARD"}
        np.testing.assert_allclose(evaluate(recipe, nodes, leaves, mode="deployment"), p)
        self.assertEqual(active_leaf_ids(recipe, nodes), ["A", "BASE", "GUARD"])

    def test_final_guard_overrides_pool_and_bias(self):
        nodes = {
            "a": {"kind": "leaf", "leaf_id": "A"},
            "b": {"kind": "leaf", "leaf_id": "B"},
            "mix": {"kind": "mix", "sources": ["a", "b"], "weights": [.5, .5], "pooling": "arithmetic"},
            "root": {"kind": "fixed_bias", "source": "mix", "bias": [1.0, -1.0]},
        }
        base = np.asarray([[.2, .8], [.7, .3]])
        leaves = {
            "A": np.asarray([[.9, .1], [.2, .8]]), "B": np.asarray([[.8, .2], [.1, .9]]),
            "BASE": base, "GUARD": np.asarray([True, False]),
        }
        output = evaluate({"root": "root", "guard_baseline_leaf": "BASE", "guard_mask_leaf": "GUARD"}, nodes, leaves, mode="deployment")
        np.testing.assert_allclose(output[0], base[0])
        self.assertFalse(np.allclose(output[1], base[1]))

    def test_cycle_and_nonfinite_bias_fail_closed(self):
        leaves = {"BASE": np.full((2, 2), .5), "GUARD": np.zeros(2, bool)}
        recipe = {"root": "x", "guard_baseline_leaf": "BASE", "guard_mask_leaf": "GUARD"}
        with self.assertRaisesRegex(ValueError, "cycle"):
            evaluate(recipe, {"x": {"kind": "fixed_bias", "source": "x", "bias": [0, 0]}}, leaves, mode="deployment")
        with self.assertRaisesRegex(ValueError, "invalid"):
            evaluate(recipe, {
                "a": {"kind": "leaf", "leaf_id": "BASE"},
                "x": {"kind": "fixed_bias", "source": "a", "bias": [float("nan"), 0]},
            }, leaves, mode="deployment")

    def test_nan_weights_length_mismatch_and_string_guard_fail_closed(self):
        p = np.full((2, 2), .5)
        leaves = {"A": p, "BASE": p, "GUARD": np.zeros(2, bool)}
        recipe = {"root": "x", "guard_baseline_leaf": "BASE", "guard_mask_leaf": "GUARD"}
        with self.assertRaisesRegex(ValueError, "weights"):
            evaluate(recipe, {"x": {"kind": "mix", "sources": [], "weights": [1], "pooling": "arithmetic"}}, leaves, mode="deployment")
        with self.assertRaisesRegex(ValueError, "weights"):
            pool([p, p], [float("nan"), float("nan")], "arithmetic")
        with self.assertRaisesRegex(ValueError, "guard"):
            evaluate(
                {"root": "x", "guard_baseline_leaf": "BASE", "guard_mask_leaf": "BAD"},
                {"x": {"kind": "leaf", "leaf_id": "A"}},
                {**leaves, "BAD": np.asarray(["false", "false"])}, mode="deployment",
            )

    def test_strength_zero_still_validates_bias_metadata(self):
        p = np.full((2, 2), .5)
        nodes = {
            "a": {"kind": "leaf", "leaf_id": "A"},
            "x": {"kind": "crossfit_bias", "source": "a", "strength": 0.0,
                  "fold_biases": {"0": [float("nan"), 0]}, "deployment_bias": [0, 0]},
        }
        recipe = {"root": "x", "guard_baseline_leaf": "BASE", "guard_mask_leaf": "GUARD"}
        with self.assertRaisesRegex(ValueError, "crossfit bias"):
            evaluate(recipe, nodes, {"A": p, "BASE": p, "GUARD": np.zeros(2, bool)}, mode="search", fold_ids=np.zeros(2, int))

    def test_slim_v2_meta_graph_ignores_missing_zero_weight_child(self):
        p = np.asarray([[.8, .2], [.1, .9]])
        round0 = SimpleNamespace(
            recipe_id="r0", round_index=0, anchor_id="A1", weights=(1, 0, 0, 0, 0),
            slot_candidate_ids=("C0", "C1", "C2", "C3"), pooling="geometric",
        )
        meta = SimpleNamespace(
            recipe_id="r1", round_index=1, weights=(1, 0),
            slot_candidate_ids=("r0", "missing-r0"), pooling="arithmetic", bias=(.2, -.2),
        )
        translated = translate_v2_selection({"selected": meta, "candidate_graph": {"r0": round0, "r1": meta}})
        leaves = {
            "anchor:A1": p, "stage:latest": p, "mask:guard": np.zeros(2, bool),
        }
        recipe = {"root": translated["root"], "guard_baseline_leaf": "stage:latest", "guard_mask_leaf": "mask:guard"}
        output = evaluate(recipe, translated["nodes"], leaves, mode="deployment")
        logits = np.log(p) + np.asarray([.2, -.2]); logits -= logits.max(axis=1, keepdims=True)
        expected = np.exp(logits); expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(output, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
