from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import search
from combo_core import apply_bias, canonical_sha256, normalize, pool


class SearchIntegrationTests(unittest.TestCase):
    def test_nested_nonzero_crossfit_graph_matches_manual_retained_probability(self):
        left = normalize(np.asarray([[.8,.2],[.3,.7],[.6,.4],[.4,.6]]))
        right = normalize(np.asarray([[.5,.5],[.7,.3],[.2,.8],[.9,.1]]))
        baseline = left.copy(); guard = np.asarray([False, True, False, False], dtype=bool)
        folds = np.asarray([0, 0, 1, 1]); bias0 = np.asarray([.2, -.2]); bias1 = np.asarray([-.1, .1])
        pooled = pool([left, right], [.25, .75], "geometric")
        expected = np.zeros_like(pooled)
        expected[folds == 0] = apply_bias(pooled[folds == 0], bias0 * .5)
        expected[folds == 1] = apply_bias(pooled[folds == 1], bias1 * .5)
        expected[guard] = baseline[guard]
        nodes = {
            "L": {"kind": "leaf", "leaf_id": "candidate:L"},
            "R": {"kind": "leaf", "leaf_id": "candidate:R"},
            "M": {"kind": "mix", "sources": ["L", "R"], "weights": [.25, .75], "pooling": "geometric"},
            "B": {"kind": "crossfit_bias", "source": "M", "strength": .5,
                  "fold_biases": {"0": bias0.tolist(), "1": bias1.tolist()},
                  "deployment_bias": [0.0, 0.0], "bias_receipt": {"synthetic": True}},
        }
        recipe = {"root": "B", "guard_baseline_leaf": "stage:latest", "guard_mask_leaf": "mask:guard"}
        result = search._verify_frozen_recipe(
            recipe, nodes,
            {"candidate:L": left, "candidate:R": right, "stage:latest": baseline, "mask:guard": guard},
            folds, expected, np.asarray([0, 1, 1, 0]),
        )
        self.assertTrue(result["frozen_reproduction_argmax_identical"])
        self.assertLessEqual(result["frozen_reproduction_max_abs"], 1e-12)

    def test_parent_checkpoint_recovers_each_missing_counterpart_and_rejects_mismatch(self):
        rows = [{"recipe_id": "R0", "macro_f1": 0.1}]
        probabilities = {"R0": {"probability": np.asarray([[0.7, 0.3]])}}
        nodes = {"R0": {"kind": "leaf", "leaf_id": "candidate:C0"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search._save_parents(root, "C0", rows, probabilities, nodes, ["ID0"], ["A", "B"])
            probability_path = root / "parents/C0/probabilities.npz"
            recipe_path = root / "parents/C0/recipes.joblib"
            recipe_bytes = recipe_path.read_bytes()
            recipe_path.unlink()
            search._save_parents(root, "C0", rows, probabilities, nodes, ["ID0"], ["A", "B"])
            self.assertTrue(recipe_path.is_file())
            probability_path.unlink()
            search._save_parents(root, "C0", rows, probabilities, nodes, ["ID0"], ["A", "B"])
            self.assertTrue(probability_path.is_file())
            recipe_path.write_bytes(recipe_bytes + b"tamper")
            with patch.object(search.joblib, "load", side_effect=AssertionError("must not deserialize")):
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    search._save_parents(root, "C0", rows, probabilities, nodes, ["ID0"], ["A", "B"])

    def test_frozen_recipe_partial_pair_is_reused_and_mismatch_rejected(self):
        payload = {
            "recipe": {"schema_version": "FROZEN_COMBO_RECIPE_V1", "root": "N"},
            "nodes": {"N": {"kind": "leaf", "leaf_id": "stage:latest"}},
        }
        metadata = {"best_recipe_id": "BEST", "best_inner_macro_f1": 0.5, "active_node_count": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = search._save_frozen_recipe(root, payload, metadata)
            receipt_path = root / "FROZEN_RECIPE.json"
            receipt_path.unlink()
            second = search._save_frozen_recipe(root, payload, metadata)
            self.assertEqual(first, second)
            frozen_path = root / "FROZEN_RECIPE.joblib"
            frozen_path.unlink()
            third = search._save_frozen_recipe(root, payload, metadata)
            self.assertEqual(third, first)
            frozen_bytes = frozen_path.read_bytes(); frozen_path.write_bytes(frozen_bytes + b"tamper")
            with patch.object(search.joblib, "load", side_effect=AssertionError("must not deserialize")):
                with self.assertRaisesRegex(ValueError, "recipe changed"):
                    search._save_frozen_recipe(root, payload, metadata)
            frozen_path.write_bytes(frozen_bytes)
            receipt_path.write_text(json.dumps({**third, "best_recipe_id": "OTHER"}))
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                search._save_frozen_recipe(root, payload, metadata)

    def fixture(self, root):
        rows, classes = 78, 26
        y = np.tile(np.arange(classes), 3)
        rng = np.random.default_rng(42)
        raw = rng.gamma(1.5, 1, size=(rows, classes)); baseline = raw / raw.sum(1, keepdims=True)
        candidate_ids = [f"C{i:03d}" for i in range(124)]
        candidate_probabilities = []
        for index in range(124):
            value = baseline + rng.normal(0, .002 + index * 1e-6, baseline.shape)
            value = np.maximum(value, 1e-6); candidate_probabilities.append(value / value.sum(1, keepdims=True))
        evidence = root / "EVIDENCE.npz"
        with evidence.open("wb") as stream:
            np.savez_compressed(
                stream, ids=np.asarray([f"R{i}" for i in range(rows)], dtype="U"), y=y,
                groups=np.asarray([f"G{i}" for i in range(rows)], dtype="U"), outer_folds=np.tile(np.arange(3), 26),
                inner_folds=np.tile(np.arange(3), 26), class_order=np.asarray([f"K{i}" for i in range(classes)], dtype="U"),
                guard=np.asarray([i % 17 == 0 for i in range(rows)], dtype=bool),
                candidate_ids=np.asarray(candidate_ids, dtype="U"),
                candidate_families=np.asarray([f"F{i % 12}" for i in range(124)], dtype="U"),
                candidate_probabilities=np.stack(candidate_probabilities),
                anchor_ids=np.asarray(["A1_GEOMETRIC_GUARD", "A2_C10_BIAS_GEOMETRIC_GUARD"], dtype="U"),
                anchor_probabilities=np.stack([baseline, baseline]), stage_probability=baseline,
            )
        with np.load(evidence, allow_pickle=False) as stored:
            receipt = {"evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(), "stage_deployment_sha256": "d" * 64}
            for name in ("ids", "y", "groups", "outer_folds", "inner_folds", "class_order"):
                receipt[f"{name}_sha256"] = canonical_sha256(np.asarray(stored[name]).tolist())
        round0 = SimpleNamespace(
            recipe_id="stage-r0", round_index=0, anchor_id="A1_GEOMETRIC_GUARD",
            slot_candidate_ids=("C000", "C001", "C002", "C003"),
            weights=(1.0, 0.0, 0.0, 0.0, 0.0), pooling="arithmetic", probability=baseline,
        )
        selection_path = root / "stage_selection.joblib"
        joblib.dump({"selected": round0, "candidate_graph": {"stage-r0": round0}}, selection_path)
        receipt["stage_selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        receipt_path = root / "EVIDENCE.json"; receipt_path.write_text(json.dumps(receipt))
        manifest = ROOT / "GRID_MANIFEST.json"
        config = {
            "evidence_npz": str(evidence), "evidence_receipt_json": str(receipt_path),
            "evidence_sha256": receipt["evidence_sha256"], "grid_manifest_json": str(manifest),
            "grid_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "stage_selection_joblib": str(selection_path), "v2_root": str(root),
            "meta_source_sha256": "m" * 64, "_runtime_config_sha256": "c" * 64,
        }
        return config

    @staticmethod
    def zero_bias(v2_root, probability, y, inner_folds, cache_root):
        classes = probability.shape[1]
        folds = {str(value): np.zeros(classes) for value in np.unique(inner_folds)}
        return folds, np.zeros(classes), {"synthetic": True}

    def test_full_grid_interruption_resume_matches_clean_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); config = self.fixture(base)
            resumed = base / "resumed"; clean = base / "clean"
            original = search.Ledger.record
            injected = {"done": False}
            def record(ledger, index, row):
                if ledger.path.name == "C1.jsonl" and index == 17 and not injected["done"]:
                    injected["done"] = True; raise RuntimeError("injected")
                return original(ledger, index, row)
            with patch.object(search, "_crossfit_biases", self.zero_bias), patch.object(search.Ledger, "record", record):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    search.run_search(config, resumed)
            with patch.object(search, "_crossfit_biases", self.zero_bias):
                resumed_result = search.run_search(config, resumed)
                clean_result = search.run_search(config, clean)
            self.assertEqual(resumed_result["best_recipe_id"], clean_result["best_recipe_id"])
            self.assertEqual(resumed_result["best_inner_macro_f1"], clean_result["best_inner_macro_f1"])
            self.assertEqual(resumed_result["declared_score_rows"], 6241)
            self.assertEqual(clean_result["declared_score_rows"], 6241)
            self.assertEqual(len((resumed / "ledgers/C0.jsonl").read_text().splitlines()), 3721)
            self.assertEqual(len((resumed / "ledgers/C1.jsonl").read_text().splitlines()), 1260)
            self.assertEqual(len((resumed / "ledgers/C2.jsonl").read_text().splitlines()), 1260)
            frozen = joblib.load(resumed / "FROZEN_RECIPE.joblib")
            self.assertIn(frozen["recipe"]["root"], frozen["nodes"])
            changed = dict(config, _runtime_config_sha256="x" * 64)
            with patch.object(search, "_crossfit_biases", self.zero_bias), self.assertRaisesRegex(ValueError, "identity|provenance"):
                search.run_search(changed, resumed)
            for name in ("C1", "C2"):
                complete = json.loads((resumed / f"ledgers/{name}.COMPLETE.json").read_text())
                self.assertEqual(complete["declared_rows"], 1260)
                self.assertLess(complete["unique_effective_rows"], 1260)

    def test_bias_fit_cache_reuses_exact_probability_identity(self):
        v2 = ROOT.parent / "staged_team_ensemble_20260916_recovery_v2"
        import combo_log_bias
        rows, classes = 78, 26
        probability = np.full((rows, classes), 1 / classes)
        y = np.tile(np.arange(classes), 3); folds = np.tile(np.arange(3), 26)
        calls = []
        def fake_fit(value, labels):
            calls.append(len(value)); return np.zeros(classes), {"rows": len(value)}
        with tempfile.TemporaryDirectory() as directory, patch.object(combo_log_bias, "fit_c10_log_bias", fake_fit):
            first = search._crossfit_biases(v2, probability, y, folds, directory)
            second = search._crossfit_biases(v2, probability, y, folds, directory)
        self.assertEqual(calls, [52, 52, 52, 78])
        for left, right in zip(first[:2], second[:2]):
            if isinstance(left, dict):
                for key in left: np.testing.assert_array_equal(left[key], right[key])
            else: np.testing.assert_array_equal(left, right)


if __name__ == "__main__": unittest.main()
