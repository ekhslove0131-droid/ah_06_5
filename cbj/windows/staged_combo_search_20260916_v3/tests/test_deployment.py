from __future__ import annotations

import hashlib
import json
import joblib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import deployment
from combo_core import apply_bias, pool
from deployment import _component_paths, _import_component, _verify_provenance, _write_outputs


def _module(**values):
    module = types.ModuleType("synthetic")
    for name, value in values.items(): setattr(module, name, value)
    return module


class DeploymentTests(unittest.TestCase):
    def test_fivefold_deploy_uses_only_active_component_and_complete_resume_skips_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); v2 = root / "v2"; v2run = root / "v2run"; search = root / "search"; evidence = root / "evidence"; output = root / "deploy"
            for path in (v2, v2run, search, evidence): path.mkdir()
            (v2 / "runtime_config.recovery.windows.json").write_text("{}")
            prepared = {"source_sha256": "v" * 64, "runtime_config_sha256": _sha_for_test(v2 / "runtime_config.recovery.windows.json")}
            (v2run / "PREPARED.json").write_text(json.dumps(prepared))
            evidence_npz = evidence / "EVIDENCE.npz"; evidence_npz.write_bytes(b"evidence")
            stage = evidence / "stage_deployment.json"; stage.write_text(json.dumps({"bias_receipt": {"centered_bias": [0.0] * 26}}))
            evidence_receipt = {"evidence_sha256": _sha_for_test(evidence_npz), "stage_deployment_sha256": _sha_for_test(stage)}
            (evidence / "EVIDENCE.json").write_text(json.dumps(evidence_receipt))
            nodes = {
                "anchor": {"kind": "leaf", "leaf_id": "anchor:A1_GEOMETRIC_GUARD"},
                "candidate": {"kind": "leaf", "leaf_id": "candidate:C1"},
                "stage": {"kind": "mix", "sources": ["anchor"], "weights": [1.0], "pooling": "arithmetic",
                          "guard_mask_leaf": "mask:guard", "guard_baseline_source": "anchor"},
                "combo": {"kind": "mix", "sources": ["stage", "candidate"], "weights": [.5, .5], "pooling": "geometric"},
            }
            recipe = {"root": "combo", "guard_baseline_leaf": "stage:latest", "guard_mask_leaf": "mask:guard", "stage_graph_root": "stage"}
            frozen_path = search / "FROZEN_RECIPE.joblib"; joblib.dump({"recipe": recipe, "nodes": nodes}, frozen_path)
            frozen_receipt = {"best_recipe_id": "BEST", "joblib_sha256": _sha_for_test(frozen_path)}
            (search / "FROZEN_RECIPE.json").write_text(json.dumps(frozen_receipt))
            search_receipt = {
                "best_recipe_id": "BEST", "meta_source_sha256": "m" * 64, "runtime_config_sha256": "c" * 64,
                "evidence_sha256": evidence_receipt["evidence_sha256"],
                "stage_deployment_sha256": evidence_receipt["stage_deployment_sha256"],
                "frozen_recipe_joblib_sha256": frozen_receipt["joblib_sha256"],
                "frozen_recipe_receipt_sha256": _sha_for_test(search / "FROZEN_RECIPE.json"),
            }
            (search / "SEARCH_COMPLETE.json").write_text(json.dumps(search_receipt))
            classes = [f"K{i}" for i in range(26)]
            data = {"ids": np.asarray([f"R{i}" for i in range(5)]), "y_int": np.arange(5),
                    "groups": np.asarray([f"G{i}" for i in range(5)]), "folds": np.arange(5),
                    "class_order": classes, "y_label": np.asarray(classes[:5])}
            test = pd.DataFrame({"ID": ["T0", "T1"]}); sample = test.assign(SUBCLASS="K0")
            counts = {"candidate": 0, "anchor": 0, "inference": 0}
            uniform_valid = np.full((1,26), 1/26); uniform_test = np.full((2,26), 1/26)
            candidate_valid = uniform_valid.copy(); candidate_valid[:,0] += .02; candidate_valid /= candidate_valid.sum(1,keepdims=True)
            candidate_test = uniform_test.copy(); candidate_test[:,0] += .02; candidate_test /= candidate_test.sum(1,keepdims=True)
            def fit_candidate(*args, **kwargs): counts["candidate"] += 1; return candidate_valid, candidate_test
            def fit_anchors(*args, **kwargs):
                counts["anchor"] += 1
                return {"H1": (uniform_valid, uniform_test), "C10": (uniform_valid, uniform_test), "guard": np.zeros(3, bool)}
            def load_inference(*args): counts["inference"] += 1; return test, sample
            partitions = {"outer": [{"fold": fold, "train_indices": [i for i in range(5) if i != fold], "valid_indices": [fold]} for fold in range(5)]}
            grade_runner = types.ModuleType("grade_runner")
            grade_runner._component_bindings = lambda name, declaration_hash, fold, *args: {"component": name, "fold": fold, "source_sha256": "v"*64}
            grade_runner._declarations = lambda root: ({}, [], {"C1": {"candidate_id": "C1", "config_hash": "1"*64}})
            grade_runner._fit_anchor_components = fit_anchors; grade_runner._fit_component = fit_candidate
            grade_runner._load_inference = load_inference; grade_runner._source_identity = lambda root: ("v"*64, [])
            modules = {
                "grade_runner": grade_runner,
                "anchor_pipeline": _module(anchor_model_recipes=lambda: {"H1": {"config_hash":"h"*64}, "C10": {"config_hash":"c"*64}}),
                "atomic_state": _module(read_json=lambda path: json.loads(Path(path).read_text())),
                "log_bias": _module(apply_log_bias=apply_bias),
                "partition_contract": _module(build_partitions=lambda *args: partitions),
                "runner": _module(_runner_live=lambda root: False),
                "support_guard": _module(
                    geometric_pool=lambda left,right: pool([left,right],[.5,.5],"geometric"),
                    guarded_probability=lambda value,base,guard: np.where(np.asarray(guard)[:,None],base,value),
                ),
                "workflow": _module(_load_training=lambda config: data),
            }
            config = {"v2_root": str(v2), "v2_run_root": str(v2run), "combo_search_run_root": str(search),
                      "combo_evidence_root": str(evidence), "evidence_npz": str(evidence_npz),
                      "evidence_sha256": evidence_receipt["evidence_sha256"], "meta_source_sha256": "m"*64,
                      "_runtime_config_sha256": "c"*64}
            with patch.dict("sys.modules", modules), patch.object(deployment, "source_identity", lambda root: ("m"*64, [])):
                first = deployment.deploy(config, output)
                second = deployment.deploy(config, output)
            self.assertEqual(counts, {"candidate": 5, "anchor": 5, "inference": 1})
            self.assertEqual(first, second)
            self.assertEqual(first["selected_components"], ["C1"])
            with np.load(output / "predictions.npz", allow_pickle=False) as stored:
                self.assertEqual(stored["folds"].tolist(), list(range(5)))
                self.assertEqual(stored["test_ids"].dtype.kind, "U")
            search_receipt["frozen_recipe_receipt_sha256"] = "0" * 64
            (search / "SEARCH_COMPLETE.json").write_text(json.dumps(search_receipt))
            with patch.dict("sys.modules", modules), patch.object(deployment, "source_identity", lambda root: ("m"*64, [])):
                with self.assertRaisesRegex(ValueError, "completed search/frozen recipe binding"):
                    deployment.deploy(config, output)
            self.assertEqual(counts["inference"], 1)

    def test_provenance_tamper_rejected_before_inference_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); v2 = root / "v2"; search = root / "search"; evidence = root / "evidence"
            v2.mkdir(); search.mkdir(); evidence.mkdir()
            v2_config = v2 / "runtime_config.recovery.windows.json"; v2_config.write_text("{}")
            evidence_npz = evidence / "EVIDENCE.npz"; evidence_npz.write_bytes(b"evidence")
            stage = evidence / "stage_deployment.json"; stage.write_text("{}")
            evidence_sha = hashlib.sha256(evidence_npz.read_bytes()).hexdigest(); stage_sha = hashlib.sha256(stage.read_bytes()).hexdigest()
            (evidence / "EVIDENCE.json").write_text(json.dumps({"evidence_sha256": evidence_sha, "stage_deployment_sha256": stage_sha}))
            (search / "SEARCH_COMPLETE.json").write_text(json.dumps({
                "meta_source_sha256": "m" * 64, "runtime_config_sha256": "c" * 64,
                "evidence_sha256": evidence_sha, "stage_deployment_sha256": stage_sha,
            }))
            config = {
                "combo_search_run_root": str(search), "combo_evidence_root": str(evidence),
                "evidence_npz": str(evidence_npz), "evidence_sha256": evidence_sha,
                "meta_source_sha256": "m" * 64, "_runtime_config_sha256": "c" * 64,
            }
            prepared = {"source_sha256": "v" * 64, "runtime_config_sha256": hashlib.sha256(v2_config.read_bytes()).hexdigest()}
            _verify_provenance(config, v2, prepared, "v" * 64, "m" * 64)
            stage.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "bias evidence"):
                _verify_provenance(config, v2, prepared, "v" * 64, "m" * 64)

    def test_component_cache_import_skips_fit_surface_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / "v2"; target = base / "new"
            npz, receipt = _component_paths(source, "C1", 0); npz.parent.mkdir(parents=True)
            with npz.open("wb") as stream:
                np.savez_compressed(stream, valid=np.full((2, 3), 1 / 3), test=np.full((4, 3), 1 / 3))
            bindings = {"component": "C1", "fold": 0, "source_sha256": "a" * 64}
            value = {"bindings": bindings, "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
            receipt.write_text(json.dumps(value))
            self.assertTrue(_import_component(source, target, "C1", 0, bindings))
            imported_npz, imported_receipt = _component_paths(target, "C1", 0)
            self.assertEqual(imported_npz.read_bytes(), npz.read_bytes())
            self.assertEqual(json.loads(imported_receipt.read_text()), value)
            with tempfile.TemporaryDirectory() as other:
                npz.write_bytes(npz.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "lineage"):
                    _import_component(source, other, "C1", 0, bindings)

    def test_output_roundtrip_uses_unicode_ids_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); classes = ["A", "B", "C"]
            data = {
                "ids": np.asarray([f"R{i}" for i in range(10)]), "y_int": np.arange(10) % 3,
                "groups": np.asarray([f"G{i}" for i in range(10)]), "folds": np.arange(10) % 5,
                "class_order": classes,
            }
            test = pd.DataFrame({"ID": ["T가", "T나", "T다"]}); sample = test.assign(SUBCLASS="A")
            oof = np.full((10, 3), 1 / 3); test_probability = np.asarray([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
            complete = _write_outputs(root, data, test, sample, oof, test_probability, {"status": "COMPLETE"})
            with np.load(root / "predictions.npz", allow_pickle=False) as stored:
                self.assertEqual(stored["ids"].dtype.kind, "U")
                self.assertEqual(stored["test_ids"].dtype.kind, "U")
                self.assertEqual(stored["test_ids"].tolist(), test["ID"].tolist())
                np.testing.assert_allclose(stored["oof_probability"].sum(1), 1)
            output = pd.read_csv(root / "submission_combo_frozen.csv")
            self.assertEqual(output["ID"].tolist(), test["ID"].tolist())
            self.assertEqual(output["SUBCLASS"].tolist(), classes)
            self.assertEqual(len(complete["predictions_sha256"]), 64)
            replay = _write_outputs(root, data, test, sample, oof, test_probability, {"status": "COMPLETE"})
            self.assertEqual(replay["predictions_sha256"], complete["predictions_sha256"])

    def test_output_resume_after_csv_boundary_writes_missing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); classes = ["A", "B"]
            data = {"ids": np.asarray([f"R{i}" for i in range(5)]), "y_int": np.arange(5) % 2,
                    "groups": np.asarray([f"G{i}" for i in range(5)]), "folds": np.arange(5),
                    "class_order": classes}
            test = pd.DataFrame({"ID": ["T0", "T1"]}); sample = test.assign(SUBCLASS="A")
            oof = np.full((5,2), .5); probability = np.asarray([[.9,.1],[.2,.8]])
            expected = sample.copy(); expected["SUBCLASS"] = classes
            expected.to_csv(root / "submission_combo_frozen.csv", index=False)
            complete = _write_outputs(root, data, test, sample, oof, probability, {"status": "COMPLETE"})
            self.assertTrue((root / "predictions.npz").is_file())
            self.assertEqual(_sha_for_test(root / "submission_combo_frozen.csv"), complete["submission_sha256"])


def _sha_for_test(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__": unittest.main()
