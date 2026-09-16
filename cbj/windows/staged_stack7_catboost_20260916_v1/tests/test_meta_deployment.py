import hashlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import baseline
import meta_deployment
from meta_search import run_search
from test_artifact_store import _bank_fixture
from test_meta_search import _baseline_fixture


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _probability_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _normalized(rows, classes, offset):
    value = np.empty((rows, classes), dtype=np.float64)
    for row in range(rows):
        raw = np.asarray([1.0 + ((row + offset + col) % classes) for col in range(classes)])
        value[row] = raw / raw.sum()
    return value


def _baseline_context(root):
    root = Path(root)
    source = root / "v3_source"; source.mkdir()
    source_file = source / "combo_core.py"
    source_file.write_text(textwrap.dedent("""
        import numpy as np
        def normalize(value):
            value = np.asarray(value, dtype=np.float64)
            return value / value.sum(axis=1, keepdims=True)
        def probability_sha256(value):
            import hashlib
            return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()
        def evaluate(recipe, nodes, leaves, *, mode, fold_ids=None):
            def visit(node_id):
                node = nodes[node_id]
                if node['kind'] == 'leaf': return normalize(leaves[node['leaf_id']])
                raise ValueError('fixture graph kind')
            return visit(recipe['root'])
    """))
    (source / "SOURCE_ALLOWLIST.txt").write_text("combo_core.py\n")
    source_rows = [{"path": "combo_core.py", "sha256": _sha(source_file)}]
    source_sha = hashlib.sha256(_json_bytes(source_rows)).hexdigest()
    runtime = root / "runtime.json"; runtime.write_bytes(_json_bytes({"meta_source_sha256": source_sha}))
    search = root / "v3_search"; search.mkdir()
    frozen = {"recipe": {"root": "FINAL", "stage_graph_root": "STAGE"}, "nodes": {
        "STAGE": {"kind": "leaf", "leaf_id": "anchor:A2_C10_BIAS_GEOMETRIC_GUARD"},
        "FINAL": {"kind": "leaf", "leaf_id": "stage:latest"},
    }}
    frozen_path = search / "FROZEN_RECIPE.joblib"; joblib.dump(frozen, frozen_path)
    frozen_receipt = {"best_recipe_id": "FINAL", "joblib_sha256": _sha(frozen_path)}
    frozen_receipt_path = search / "FROZEN_RECIPE.json"; frozen_receipt_path.write_bytes(_json_bytes(frozen_receipt))
    search_receipt = {
        "status": "COMPLETE", "best_recipe_id": "FINAL", "frozen_recipe_joblib_sha256": _sha(frozen_path),
        "frozen_recipe_receipt_sha256": _sha(frozen_receipt_path), "meta_source_sha256": source_sha,
        "runtime_config_sha256": _sha(runtime), "test_read": False, "full_oof_used": False,
    }
    (search / "SEARCH_COMPLETE.json").write_bytes(_json_bytes(search_receipt))
    stage = root / "stage_deployment.json"
    stage.write_bytes(_json_bytes({"bias_receipt": {"centered_bias": [0.25, -0.25]}}))
    return {
        "v3_source_root": str(source), "v3_runtime_config_path": str(runtime),
        "v3_search_root": str(search), "v3_source_sha256": source_sha,
        "v3_runtime_config_sha256": _sha(runtime), "v3_frozen_recipe_sha256": _sha(frozen_path),
        "v3_stage_deployment_json": str(stage), "v3_stage_deployment_sha256": _sha(stage),
    }


def _search_fixture(root, baseline_label_shift=1):
    root = Path(root)
    config = _baseline_fixture(root, _bank_fixture(root), baseline_label_shift=baseline_label_shift)
    config["fit_meta_callback"] = lambda declaration, p, y: {"declaration": declaration}
    config["predict_meta_callback"] = lambda artifact, p: p[0]
    run = root / "meta_search"
    with redirect_stdout(io.StringIO()):
        result = run_search(config, run)
    config.update({
        "meta_search_run_root": str(run),
        "meta_search_selection_sha256": result["selection_sha256"],
        "meta_search_complete_sha256": _sha(run / "SEARCH_COMPLETE.json"),
    })
    return config


def _rebind_search(run):
    run = Path(run); scores = run / "scores"; commits = run / "score_commits"
    rows = [json.loads((scores / f"{i:04d}.json").read_text()) for i in range(541)]
    ledger = b"".join(_json_bytes(row) for row in rows); (run / "SCORE_LEDGER.jsonl").write_bytes(ledger)
    commit_set = hashlib.sha256(_json_bytes([
        {"declared_index": i, "commit_sha256": _sha(commits / f"{i:04d}.json"),
         "row_bytes_sha256": json.loads((commits / f"{i:04d}.json").read_text())["row_bytes_sha256"]}
        for i in range(541)
    ])).hexdigest(); ledger_sha = hashlib.sha256(ledger).hexdigest()
    score_complete = json.loads((run / "SCORE_COMPLETE.json").read_text()); score_complete.update(ledger_sha256=ledger_sha, score_commit_set_sha256=commit_set)
    (run / "SCORE_COMPLETE.json").write_bytes(_json_bytes(score_complete))
    selection = json.loads((run / "FROZEN_SELECTION.json").read_text()); selection.update(ledger_sha256=ledger_sha, score_commit_set_sha256=commit_set)
    (run / "FROZEN_SELECTION.json").write_bytes(_json_bytes(selection))
    complete = json.loads((run / "SEARCH_COMPLETE.json").read_text()); complete.update(
        selection_sha256=_sha(run / "FROZEN_SELECTION.json"), ledger_sha256=ledger_sha, score_commit_set_sha256=commit_set)
    (run / "SEARCH_COMPLETE.json").write_bytes(_json_bytes(complete))


class MetaDeploymentUnitTests(unittest.TestCase):
    def test_frozen_gate_reproduces_meta_and_incumbent_without_any_verification_fit(self):
        for label_shift, expected_deployment in ((1, True), (0, False)):
            with self.subTest(label_shift=label_shift), tempfile.TemporaryDirectory() as td:
                config = _search_fixture(td, baseline_label_shift=label_shift)
                config["fit_meta_callback"] = lambda *_: self.fail("verification attempted fit callback")
                config["predict_meta_callback"] = lambda *_: self.fail("verification attempted prediction callback")
                selection, binding = meta_deployment.verify_frozen_selection(config)
                self.assertEqual(selection["deployment_required"], expected_deployment)
                self.assertEqual(binding["selected_inner_probability_sha256"], selection["selected_probability_sha256"])

    def test_frozen_gate_rejects_caller_and_verified_inner_binding_mismatches(self):
        with tempfile.TemporaryDirectory() as td:
            config = _search_fixture(td)
            mutations = {
                "source": ("meta_source_sha256", "wrong-source"),
                "runtime": ("_runtime_config_sha256", "wrong-config"),
                "bank": ("training_bank_sha256", "0" * 64),
                "bank_receipt": ("training_bank_receipt_sha256", "1" * 64),
                "recipe": ("v3_frozen_recipe_sha256", "2" * 64),
                "baseline": ("expected_baseline_inner_macro_f1", 0.123),
            }
            for name, (key, value) in mutations.items():
                changed = dict(config); changed[key] = value
                with self.subTest(name=name), self.assertRaises(ValueError):
                    meta_deployment.verify_frozen_selection(changed)

    def test_frozen_gate_rejects_selected_cache_missing_or_tampered_before_full_test(self):
        for case in ("receipt_missing", "model_tamper", "probability_tamper", "selection_receipt_tamper"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                config = _search_fixture(td); run = Path(config["meta_search_run_root"])
                selection_path = run / "FROZEN_SELECTION.json"; selection = json.loads(selection_path.read_text())
                selected = selection["selected_fold_receipts"][0]
                receipt_path = Path(selected["receipt_path"]); target = receipt_path.parent
                if case == "receipt_missing":
                    receipt_path.unlink()
                elif case == "model_tamper":
                    (target / "model.joblib").write_bytes(b"tampered")
                elif case == "probability_tamper":
                    (target / "probability.npz").write_bytes(b"tampered")
                else:
                    selection["selected_fold_receipts"][0]["receipt_sha256"] = "f" * 64
                    selection_path.write_bytes(_json_bytes(selection))
                    complete_path = run / "SEARCH_COMPLETE.json"; complete = json.loads(complete_path.read_text())
                    complete["selection_sha256"] = _sha(selection_path); complete_path.write_bytes(_json_bytes(complete))
                    config["meta_search_selection_sha256"] = _sha(selection_path)
                    config["meta_search_complete_sha256"] = _sha(complete_path)
                with self.assertRaises(ValueError):
                    meta_deployment.verify_frozen_selection(config)

    def test_frozen_gate_reconstructs_selected_inner_oof_from_committed_caches(self):
        with tempfile.TemporaryDirectory() as td:
            config = _search_fixture(td); run = Path(config["meta_search_run_root"])
            selection_path = run / "FROZEN_SELECTION.json"; selection = json.loads(selection_path.read_text())
            selected = selection["selected_fold_receipts"][0]; receipt_path = Path(selected["receipt_path"])
            probability_path = receipt_path.parent / "probability.npz"
            with np.load(probability_path, allow_pickle=False) as stored:
                probability = np.asarray(stored["probability"]).copy()
            probability[:, [0, 1]] = probability[:, [1, 0]]
            np.savez_compressed(probability_path, probability=probability)
            receipt = json.loads(receipt_path.read_text())
            receipt["probability_npz_sha256"] = _sha(probability_path)
            receipt["valid_probability_sha256"] = _probability_sha(probability)
            receipt_path.write_bytes(_json_bytes(receipt))
            new_receipt_sha = _sha(receipt_path)
            for item in selection["selected_fold_receipts"]:
                if item["fold"] == selected["fold"]:
                    item["receipt_sha256"] = new_receipt_sha
                    item["valid_probability_sha256"] = receipt["valid_probability_sha256"]
            selection_path.write_bytes(_json_bytes(selection))
            for row_path in sorted((run / "scores").glob("*.json")):
                row = json.loads(row_path.read_text())
                if row.get("head_id") != selection["selected_head_id"]:
                    continue
                for item in row["fold_receipts"]:
                    if item["fold"] == selected["fold"]:
                        item["receipt_sha256"] = new_receipt_sha
                        item["valid_probability_sha256"] = receipt["valid_probability_sha256"]
                row.pop("row_sha256"); row["row_sha256"] = hashlib.sha256(_json_bytes(row)).hexdigest()
                payload = _json_bytes(row); row_path.write_bytes(payload)
                commit_path = run / "score_commits" / row_path.name; commit = json.loads(commit_path.read_text())
                commit["row_bytes_sha256"] = hashlib.sha256(payload).hexdigest(); commit["row_sha256"] = row["row_sha256"]
                commit_path.write_bytes(_json_bytes(commit))
            _rebind_search(run)
            config["meta_search_selection_sha256"] = _sha(selection_path)
            config["meta_search_complete_sha256"] = _sha(run / "SEARCH_COMPLETE.json")
            with self.assertRaisesRegex(ValueError, "selected inner OOF"):
                meta_deployment.verify_frozen_selection(config)

    def test_run_deployment_fills_five_folds_once_and_fits_only_six_meta_heads(self):
        with tempfile.TemporaryDirectory() as td:
            rows, classes, test_rows = 15, 3, 4
            folds = np.asarray([0, 1, 2, 3, 4] * 3)
            y = np.asarray([0] * 5 + [1] * 5 + [2] * 5); y[10] = 1
            ids = np.asarray([f"I{i}" for i in range(rows)]); groups = np.asarray([f"G{i}" for i in range(rows)])
            full_p = np.stack([_normalized(rows, classes, model) for model in range(7)])
            inner_mask = folds != 0
            inner_p = full_p[:, inner_mask].copy(); inner_p[0] = np.roll(inner_p[0], 1, axis=1)
            bank = {"model_ids": np.asarray(meta_deployment.MODEL_ORDER), "class_order": np.asarray(["A", "B", "C"]),
                    "full_ids": ids, "full_y": y, "full_groups": groups, "full_outer_folds": folds, "full_oof_probability": full_p}
            inner = {"model_ids": bank["model_ids"], "class_order": bank["class_order"], "inner_ids": ids[inner_mask],
                     "inner_y": y[inner_mask], "inner_groups": groups[inner_mask], "inner_outer_folds": folds[inner_mask],
                     "inner_folds": np.asarray([0, 1, 2] * 4), "inner_probability": inner_p}
            test_ids = np.asarray([f"T{i}" for i in range(test_rows)])
            components = {}
            for fold in range(5):
                valid = folds == fold; values = {}
                for model_index, model in enumerate(meta_deployment.MODEL_ORDER):
                    values[model] = {"valid": full_p[model_index, valid], "test": _normalized(test_rows, classes, model_index),
                                     "guard": np.zeros(valid.sum() + test_rows, dtype=bool) if model == "ANCHOR-C10" else None}
                components[fold] = values
            saved_oof = full_p[0].copy(); saved_test = sum(components[f]["ANCHOR-H1"]["test"] for f in range(5)) / 5
            selection = {"deployment_required": True, "selected_declaration": {"representation": "probability", "C": 0.1, "weighting": "none"},
                         "selected_alpha": 0.5, "selected_pooling": "arithmetic", "selected_head_id": "META-00", "selected_inner_macro_f1": 0.2}
            calls = {"fit": 0, "fit_rows": []}
            def fake_fit(declaration, p, labels): calls["fit"] += 1; calls["fit_rows"].append(p.shape[1]); return {"declaration": declaration}
            fake_predict = lambda artifact, p: p[0]
            originals = (meta_deployment.verify_frozen_selection, meta_deployment.load_full_bank, meta_deployment.load_inner_bank,
                         meta_deployment._load_sample, meta_deployment._load_components, meta_deployment._load_saved_v3, meta_deployment.replay_v3_fold)
            meta_deployment.verify_frozen_selection = lambda _: (selection, {"selection_sha256": "s", "search_complete_sha256": "c", "score_commit_set_sha256": "k", "ledger_sha256": "l"})
            meta_deployment.load_full_bank = lambda _: bank; meta_deployment.load_inner_bank = lambda _: inner
            meta_deployment._load_sample = lambda _: test_ids; meta_deployment._load_components = lambda *_: (components, [])
            meta_deployment._load_saved_v3 = lambda *_: (saved_oof, saved_test, {"base_source_sha256": "base"})
            meta_deployment.replay_v3_fold = lambda config, raw, guard: raw["ANCHOR-H1"]
            config = {"expected_classes": classes, "meta_source_sha256": "source", "_runtime_config_sha256": "config",
                      "training_bank_sha256": "bank", "training_bank_receipt_sha256": "receipt", "v3_frozen_recipe_sha256": "recipe",
                      "sample_submission_sha256": "sample", "v3_predictions_sha256": "pred", "v3_deployment_complete_sha256": "complete",
                      "v3_source_sha256": "v3", "fit_meta_callback": fake_fit, "predict_meta_callback": fake_predict}
            try:
                complete = meta_deployment.run_deployment(config, Path(td) / "deploy")
            finally:
                (meta_deployment.verify_frozen_selection, meta_deployment.load_full_bank, meta_deployment.load_inner_bank,
                 meta_deployment._load_sample, meta_deployment._load_components, meta_deployment._load_saved_v3, meta_deployment.replay_v3_fold) = originals
            self.assertEqual(calls["fit"], 6)
            self.assertEqual(calls["fit_rows"], [12, 12, 12, 12, 12, 12])
            self.assertEqual(complete["fit_counts"], {"base": 0, "full_meta": 5, "confirmation_meta": 1})
            with np.load(Path(td) / "deploy/predictions.npz", allow_pickle=False) as stored:
                self.assertEqual(stored["oof_probability"].shape, (15, 3)); self.assertEqual(stored["test_ids"].dtype.kind, "U")

    def test_confirmation_common_hashes_inner_fit_folds_not_outer_fold_labels(self):
        bank = {
            "full_ids": np.asarray(["V0", "F0", "F1"]), "full_y": np.asarray([0, 0, 1]),
            "full_groups": np.asarray(["VG", "G0", "G1"]), "full_outer_folds": np.asarray([0, 1, 2]),
            "model_ids": np.asarray([f"M{i}" for i in range(7)]), "class_order": np.asarray(["A", "B"]),
        }
        inner = {"inner_ids": np.asarray(["F0", "F1"]), "inner_y": np.asarray([0, 1]),
                 "inner_groups": np.asarray(["G0", "G1"]), "inner_folds": np.asarray([0, 1])}
        config = {"meta_source_sha256": "s", "_runtime_config_sha256": "c",
                  "training_bank_sha256": "b", "v3_frozen_recipe_sha256": "r"}
        declaration = {"representation": "probability", "C": 1.0, "weighting": "none"}
        common = meta_deployment._confirmation_common(config, bank, inner, declaration, bank["full_outer_folds"] == 0)
        self.assertEqual(common["fit_folds_sha256"], meta_deployment.sequence_sha256(inner["inner_folds"]))
        self.assertNotEqual(common["fit_folds_sha256"], meta_deployment.sequence_sha256(bank["full_outer_folds"][bank["full_outer_folds"] != 0]))

    def test_missing_component_cache_reports_exact_model_and_fold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); receipt_path = root / "bank.json"
            lineage = [{"model": model, "fold": fold, "npz_sha256": "a" * 64, "receipt_sha256": "b" * 64}
                       for fold in range(5) for model in meta_deployment.MODEL_ORDER]
            receipt_path.write_bytes(_json_bytes({"lineage": {"component_receipts": lineage, "base_source_sha256": "synthetic"}}))
            bank = {"class_order": np.asarray(["A", "B"]), "full_ids": np.asarray([f"I{i}" for i in range(10)]),
                    "full_y": np.asarray([0, 1] * 5), "full_groups": np.asarray([f"G{i}" for i in range(10)]),
                    "full_outer_folds": np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
                    "full_oof_probability": np.tile(np.asarray([[[0.6, 0.4]]]), (7, 10, 1))}
            config = {"training_bank_receipt_json": str(receipt_path), "v3_deployment_run_root": str(root), "expected_classes": 2}
            with self.assertRaisesRegex(ValueError, "missing component cache: ANCHOR-H1 fold 0"):
                meta_deployment._load_components(config, bank, np.asarray(["T0"]))

    def test_frozen_gate_rejects_rebound_alpha_one_alias_semantic_change(self):
        with tempfile.TemporaryDirectory() as td:
            config = _search_fixture(td); run = Path(config["meta_search_run_root"])
            alias_path = run / "scores/0010.json"; alias = json.loads(alias_path.read_text())
            alias["macro_f1"] = 0.2; alias.pop("row_sha256"); alias["row_sha256"] = hashlib.sha256(_json_bytes(alias)).hexdigest()
            payload = _json_bytes(alias); alias_path.write_bytes(payload)
            commit_path = run / "score_commits/0010.json"; commit = json.loads(commit_path.read_text())
            commit.update(row_bytes_sha256=hashlib.sha256(payload).hexdigest(), row_sha256=alias["row_sha256"])
            commit_path.write_bytes(_json_bytes(commit)); _rebind_search(run)
            config.update(meta_search_selection_sha256=_sha(run / "FROZEN_SELECTION.json"), meta_search_complete_sha256=_sha(run / "SEARCH_COMPLETE.json"))
            with self.assertRaisesRegex(ValueError, "alias"):
                meta_deployment.verify_frozen_selection(config)

    def test_selection_failure_precedes_all_full_and_test_loads(self):
        with tempfile.TemporaryDirectory() as td:
            config = _search_fixture(td); (Path(config["meta_search_run_root"]) / "scores/0000.json").unlink()
            original_full, original_sample = meta_deployment.load_full_bank, meta_deployment._load_sample
            meta_deployment.load_full_bank = lambda *_: self.fail("full bank loaded before selection verification")
            meta_deployment._load_sample = lambda *_: self.fail("test IDs loaded before selection verification")
            try:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    meta_deployment.run_deployment(config, Path(td) / "deploy")
            finally:
                meta_deployment.load_full_bank, meta_deployment._load_sample = original_full, original_sample

    def test_replay_v3_fold_uses_original_a2_bias_then_floor_1e300_geometric_guard(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_context(td)
            h1_valid = np.asarray([[1.0, 0.0], [0.4, 0.6]])
            c10_valid = np.asarray([[0.0, 1.0], [0.7, 0.3]])
            h1_test = np.asarray([[0.2, 0.8]])
            c10_test = np.asarray([[0.8, 0.2]])
            raw_valid = {name: _normalized(2, 2, index) for index, name in enumerate(meta_deployment.MODEL_ORDER)}
            raw_test = {name: _normalized(1, 2, index) for index, name in enumerate(meta_deployment.MODEL_ORDER)}
            raw_valid["ANCHOR-H1"], raw_valid["ANCHOR-C10"] = h1_valid, c10_valid
            raw_test["ANCHOR-H1"], raw_test["ANCHOR-C10"] = h1_test, c10_test
            valid = baseline.replay_v3_fold(config, raw_valid, np.asarray([False, True]))
            test = baseline.replay_v3_fold(config, raw_test, np.asarray([False]))
            bias = np.asarray([0.25, -0.25])
            def literal(h1, c10, guard):
                h1 = h1 / h1.sum(1, keepdims=True); c10 = c10 / c10.sum(1, keepdims=True)
                logits = np.log(np.maximum(c10, 1e-300)) + bias
                logits -= logits.max(1, keepdims=True); biased = np.exp(logits); biased /= biased.sum(1, keepdims=True)
                pooled_log = (np.log(np.maximum(h1, 1e-300)) + np.log(np.maximum(biased, 1e-300))) / 2
                pooled_log -= pooled_log.max(1, keepdims=True); out = np.exp(pooled_log); out /= out.sum(1, keepdims=True)
                out[guard] = h1[guard]
                return out
            np.testing.assert_allclose(valid, literal(h1_valid, c10_valid, np.asarray([False, True])), rtol=0, atol=1e-15)
            np.testing.assert_allclose(test, literal(h1_test, c10_test, np.asarray([False])), rtol=0, atol=1e-15)

    def test_full_and_confirmation_cache_identities_differ_for_same_ids_and_labels(self):
        common = {
            "source_sha256": "s", "runtime_config_sha256": "c", "bank_sha256": "b",
            "recipe_sha256": "r", "declaration_sha256": "d", "fold": 0,
            "fit_ids_sha256": "fi", "fit_y_sha256": "fy", "fit_groups_sha256": "fg",
            "fit_folds_sha256": "ff", "valid_ids_sha256": "vi", "valid_y_sha256": "vy",
            "valid_groups_sha256": "vg", "valid_folds_sha256": "vf",
            "model_order": [f"M{i}" for i in range(7)], "class_order": ["A", "B"],
        }
        full_fit = np.tile(np.asarray([[[0.8, 0.2], [0.2, 0.8]]]), (7, 1, 1))
        confirmation_fit = full_fit.copy(); confirmation_fit[0, 0] = [0.6, 0.4]
        valid = np.tile(np.asarray([[[0.3, 0.7]]]), (7, 1, 1))
        first = meta_deployment.deployment_fit_identity(common, "full_meta_outer_cv", full_fit, valid)
        second = meta_deployment.deployment_fit_identity(common, "outer0_confirmation_inner_oof", confirmation_fit, valid)
        self.assertNotEqual(meta_deployment.canonical_sha256(first), meta_deployment.canonical_sha256(second))
        self.assertEqual(first["feature_role"], "full_meta_outer_cv")
        self.assertEqual(second["feature_role"], "outer0_confirmation_inner_oof")
        self.assertEqual(first["valid_probability_tensor_sha256"], second["valid_probability_tensor_sha256"])
        self.assertNotEqual(first["fit_probability_tensor_sha256"], second["fit_probability_tensor_sha256"])

    def test_deployment_cache_reuses_sixth_head_and_rejects_corrupt_model_before_unpickle(self):
        with tempfile.TemporaryDirectory() as td:
            common = {
                "source_sha256": "s", "runtime_config_sha256": "c", "bank_sha256": "b",
                "recipe_sha256": "r", "declaration_sha256": "d", "fold": 0,
                "fit_ids_sha256": "fi", "fit_y_sha256": "fy", "fit_groups_sha256": "fg",
                "fit_folds_sha256": "ff", "valid_ids_sha256": "vi", "valid_y_sha256": "vy",
                "valid_groups_sha256": "vg", "valid_folds_sha256": "vf",
                "model_order": [f"M{i}" for i in range(7)], "class_order": ["A", "B"],
            }
            tensor = np.tile(np.asarray([[[0.8, 0.2]]]), (7, 1, 1))
            identity = meta_deployment.deployment_fit_identity(common, "full_meta_outer_cv", tensor, tensor)
            calls = {"fit": 0}
            def fit():
                calls["fit"] += 1
                return {"ok": True}
            predict = lambda model: np.asarray([[0.4, 0.6]])
            first = meta_deployment.load_or_fit_deployment_head(td, identity, fit, predict)
            second = meta_deployment.load_or_fit_deployment_head(td, identity, fit, predict)
            self.assertEqual(calls["fit"], 1)
            self.assertEqual(first[2]["identity_sha256"], second[2]["identity_sha256"])
            model_path = Path(first[2]["model_path"]); model_path.write_bytes(b"corrupt")
            original = joblib.load
            joblib.load = lambda *_args, **_kwargs: self.fail("unpickle attempted before hash verification")
            try:
                with self.assertRaisesRegex(ValueError, "model hash"):
                    meta_deployment.load_or_fit_deployment_head(td, identity, fit, predict)
            finally:
                joblib.load = original

    def test_output_publication_resumes_after_npz_and_rejects_csv_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            arrays = {
                "ids": np.asarray(["I0", "I1"], dtype="U"), "y": np.asarray([0, 1]),
                "groups": np.asarray(["G0", "G1"], dtype="U"), "folds": np.asarray([0, 1]),
                "class_order": np.asarray(["A", "B"], dtype="U"),
                "model_order": np.asarray([f"M{i}" for i in range(7)], dtype="U"),
                "oof_probability": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
                "baseline_oof_probability": np.asarray([[0.7, 0.3], [0.3, 0.7]]),
                "test_ids": np.asarray(["T0"], dtype="U"), "test_probability": np.asarray([[0.3, 0.7]]),
                "baseline_test_probability": np.asarray([[0.4, 0.6]]),
            }
            receipt = {"schema_version": "STACK7_META_DEPLOYMENT_COMPLETE_V1", "status": "COMPLETE", "selected_recipe": {"alpha": 0.5}}
            def interrupt(stage):
                if stage == "npz_published": raise RuntimeError("after npz")
            with self.assertRaisesRegex(RuntimeError, "after npz"):
                meta_deployment.publish_outputs(root, arrays, receipt, progress_callback=interrupt)
            complete = meta_deployment.publish_outputs(root, arrays, receipt)
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(np.load(root / "predictions.npz", allow_pickle=False)["test_ids"].dtype.kind, "U")
            (root / "submission_stack7_frozen.csv").write_text("ID,SUBCLASS\nT0,A\n")
            with self.assertRaisesRegex(ValueError, "submission hash"):
                meta_deployment.publish_outputs(root, arrays, receipt)


if __name__ == "__main__":
    unittest.main()
