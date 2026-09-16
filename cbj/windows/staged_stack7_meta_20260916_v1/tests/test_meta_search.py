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

from baseline import load_baseline_inner
import meta_search as meta_search_module
from meta_search import run_search

from test_artifact_store import (
    _bank_fixture,
    _canonical,
    _json_bytes,
    _sha,
)


def _macro(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p.argmax(1), labels=np.arange(p.shape[1]), average="macro", zero_division=0))


def _baseline_fixture(root, bank_config, *, baseline_label_shift=0):
    root = Path(root)
    source = root / "v3_source"; source.mkdir()
    evaluator = source / "combo_core.py"
    evaluator.write_text(textwrap.dedent("""
        import hashlib
        import numpy as np
        def probability_sha256(value):
            return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()
        def normalize(value):
            value = np.asarray(value, dtype=np.float64)
            return value / value.sum(axis=1, keepdims=True)
        def evaluate(recipe, nodes, leaves, *, mode, fold_ids=None):
            return normalize(leaves[nodes[recipe['root']]['leaf_id']])
    """))
    (source / "SOURCE_ALLOWLIST.txt").write_text("combo_core.py\n")
    rows = 12; classes = 3
    baseline_raw = np.full((rows, classes), 0.1, dtype=np.float64)
    y = np.asarray([0, 1, 2] * 4, dtype=np.int64)
    baseline_raw[np.arange(rows), (y + baseline_label_shift) % classes] = 0.8
    baseline_raw[:, 2] += 1e-16
    normalized_once = baseline_raw / baseline_raw.sum(axis=1, keepdims=True)
    baseline = normalized_once / normalized_once.sum(axis=1, keepdims=True)
    guard = np.zeros(rows, dtype=bool); guard[0] = True
    evidence = root / "EVIDENCE.npz"
    np.savez(
        evidence,
        ids=np.asarray([f"I{i}" for i in range(rows)]), y=y,
        groups=np.asarray([f"G{i}" for i in range(rows)]),
        outer_folds=np.asarray([1, 2, 3, 4] * 3, dtype=np.int16),
        inner_folds=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int16),
        class_order=np.asarray(["A", "B", "C"]), guard=guard,
        candidate_ids=np.asarray([], dtype="U1"),
        candidate_probabilities=np.empty((0, rows, classes)),
        anchor_ids=np.asarray([], dtype="U1"),
        anchor_probabilities=np.empty((0, rows, classes)),
        stage_probability=baseline_raw,
    )
    stage = root / "stage_deployment.json"; stage.write_text("{}\n")
    evidence_receipt = {
        "evidence_sha256": _sha(evidence),
        "stage_deployment_sha256": _sha(stage),
        **{f"{name}_sha256": _canonical(value.tolist()) for name, value in {
            "ids": np.asarray([f"I{i}" for i in range(rows)]), "y": y,
            "groups": np.asarray([f"G{i}" for i in range(rows)]),
            "outer_folds": np.asarray([1, 2, 3, 4] * 3, dtype=np.int16),
            "inner_folds": np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int16),
            "class_order": np.asarray(["A", "B", "C"]),
        }.items()},
    }
    evidence_receipt_path = root / "EVIDENCE.json"; evidence_receipt_path.write_bytes(_json_bytes(evidence_receipt))
    runtime = source / "runtime.json"
    source_rows = [{"path": "combo_core.py", "sha256": _sha(evaluator)}]
    source_sha = hashlib.sha256(_json_bytes(source_rows)).hexdigest()
    runtime.write_bytes(_json_bytes({"meta_source_sha256": source_sha}))
    search_root = root / "v3_search"; search_root.mkdir()
    frozen_path = search_root / "FROZEN_RECIPE.joblib"
    joblib.dump({"recipe": {"root": "ROOT"}, "nodes": {"ROOT": {"kind": "leaf", "leaf_id": "stage:latest"}}}, frozen_path)
    probability_sha = hashlib.sha256(np.ascontiguousarray(baseline, dtype="<f8").tobytes()).hexdigest()
    frozen_receipt = {"best_recipe_id": "ROOT", "joblib_sha256": _sha(frozen_path), "frozen_reproduced_probability_sha256": probability_sha}
    frozen_receipt_path = search_root / "FROZEN_RECIPE.json"; frozen_receipt_path.write_bytes(_json_bytes(frozen_receipt))
    search = {
        "status": "COMPLETE", "best_recipe_id": "ROOT", "frozen_recipe_joblib_sha256": _sha(frozen_path),
        "frozen_recipe_receipt_sha256": _sha(frozen_receipt_path), "meta_source_sha256": source_sha,
        "runtime_config_sha256": _sha(runtime), "best_inner_macro_f1": _macro(y, baseline),
        "evidence_sha256": _sha(evidence), "stage_deployment_sha256": _sha(stage),
        "test_read": False, "full_oof_used": False,
    }
    (search_root / "SEARCH_COMPLETE.json").write_bytes(_json_bytes(search))
    return {
        **bank_config,
        "evidence_npz": str(evidence), "evidence_receipt_json": str(evidence_receipt_path),
        "evidence_sha256": _sha(evidence), "expected_baseline_inner_macro_f1": _macro(y, baseline),
        "v3_source_root": str(source), "v3_runtime_config_path": str(runtime),
        "v3_search_root": str(search_root), "v3_source_sha256": source_sha,
        "v3_runtime_config_sha256": _sha(runtime), "v3_frozen_recipe_sha256": _sha(frozen_path),
        "meta_source_sha256": "meta-source", "_runtime_config_sha256": "meta-config",
        "_expected_original_probability_sha256": probability_sha,
        "_single_normalization_probability_sha256": hashlib.sha256(
            np.ascontiguousarray(normalized_once, dtype="<f8").tobytes()
        ).hexdigest(),
    }


class MetaSearchTests(unittest.TestCase):
    def test_baseline_replays_pinned_inner_graph_with_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            probability, guard, lineage = load_baseline_inner(config)
            self.assertEqual(_macro(np.asarray([0, 1, 2] * 4), probability), 1.0)
            self.assertEqual(guard.dtype, np.dtype(bool))
            self.assertTrue(guard[0])
            self.assertEqual(lineage["baseline_probability_sha256"], hashlib.sha256(np.ascontiguousarray(probability, dtype="<f8").tobytes()).hexdigest())
            self.assertEqual(lineage["baseline_probability_sha256"], config["_expected_original_probability_sha256"])
            self.assertNotEqual(
                lineage["baseline_probability_sha256"],
                config["_single_normalization_probability_sha256"],
            )

    def test_complete_search_writes_all_fits_rows_and_preserves_baseline_ties(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td, forbidden_full=True))
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_search(config, Path(td) / "run")
            self.assertEqual(result["declared_meta_heads"], 54)
            self.assertEqual(result["completed_fit_identities"], 162)
            self.assertEqual(result["declared_score_rows"], 541)
            self.assertTrue(result["winner_is_incumbent"])
            self.assertFalse(result["deployment_required"])
            self.assertEqual(len(list((Path(td) / "run" / "scores").glob("*.json"))), 541)
            rows = [json.loads(line) for line in (Path(td) / "run" / "SCORE_LEDGER.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 541)
            self.assertEqual(sum(row["kind"] == "meta_blend" for row in rows), 540)
            self.assertEqual(rows[0]["kind"], "baseline_control")
            canonical_endpoints = {
                row["head_id"]: row
                for row in rows
                if row.get("alpha") == 1.0 and row.get("pooling") == "arithmetic"
            }
            aliases = [row for row in rows if row.get("alias_of") is not None]
            self.assertEqual(len(canonical_endpoints), 54)
            self.assertEqual(len(aliases), 54)
            for alias in aliases:
                canonical = canonical_endpoints[alias["head_id"]]
                self.assertEqual(alias["pooling"], "geometric")
                self.assertEqual(alias["alias_of"], canonical["declared_index"])
                self.assertEqual(alias["endpoint_canonical_index"], canonical["declared_index"])
                self.assertEqual(alias["endpoint_probability_sha256"], canonical["probability_sha256"])
                self.assertEqual(alias["probability_sha256"], canonical["probability_sha256"])
                self.assertEqual(alias["macro_f1"], canonical["macro_f1"])
                self.assertEqual(alias["per_class_f1"], canonical["per_class_f1"])
            log = output.getvalue()
            self.assertIn("head=1/54 fold=1/3 stage=start", log)
            self.assertIn("head=1/54 fold=1/3 stage=complete cache=new", log)
            self.assertIn("head=54/54 stage=complete standalone=", log)
            self.assertIn("stage=complete rows=541 fits=162", log)

    def test_interrupted_fit_and_score_resume_without_duplicate_completed_work(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            real_fits = {"count": 0}
            def counted_fit(declaration, p, y):
                from meta_core import fit_meta
                real_fits["count"] += 1
                if real_fits["count"] == 2:
                    raise RuntimeError("interrupt after one fold")
                return fit_meta(declaration, p, y)
            config["fit_meta_callback"] = counted_fit
            with self.assertRaisesRegex(RuntimeError, "one fold"):
                run_search(config, Path(td) / "run")
            config.pop("fit_meta_callback")
            after_fit = {"count": 0}
            def resume_fit(declaration, p, y):
                from meta_core import fit_meta
                after_fit["count"] += 1
                return fit_meta(declaration, p, y)
            config["fit_meta_callback"] = resume_fit
            config["progress_callback"] = lambda event: (_ for _ in ()).throw(RuntimeError("interrupt after score")) if event["kind"] == "score" and event["declared_index"] == 540 else None
            with self.assertRaisesRegex(RuntimeError, "after score"):
                run_search(config, Path(td) / "run")
            self.assertEqual(after_fit["count"], 161)
            config.pop("progress_callback")
            final_fit = {"count": 0}
            config["fit_meta_callback"] = lambda declaration, p, y: final_fit.__setitem__("count", final_fit["count"] + 1)
            result = run_search(config, Path(td) / "run")
            self.assertEqual(final_fit["count"], 0)
            self.assertEqual(result["declared_score_rows"], 541)

    def test_corrupt_row_binding_fails_before_any_fit(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            receipt_path = Path(config["training_bank_receipt_json"])
            receipt = json.loads(receipt_path.read_text())
            receipt["array_sha256"]["inner_ids"] = "bad"
            receipt_path.write_bytes(_json_bytes(receipt))
            config["training_bank_receipt_sha256"] = _sha(receipt_path)
            calls = {"count": 0}
            config["fit_meta_callback"] = lambda *_: calls.__setitem__("count", calls["count"] + 1)
            with self.assertRaises(ValueError):
                run_search(config, Path(td) / "run")
            self.assertEqual(calls["count"], 0)

    def test_baseline_outer_fold_binding_fails_before_fit(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            evidence_path = Path(config["evidence_npz"])
            with np.load(evidence_path, allow_pickle=False) as stored:
                values = {name: np.asarray(stored[name]) for name in stored.files}
            values["outer_folds"] = np.asarray([4, 3, 2, 1] * 3, dtype=np.int16)
            np.savez(evidence_path, **values)
            evidence_receipt_path = Path(config["evidence_receipt_json"])
            receipt = json.loads(evidence_receipt_path.read_text())
            receipt["evidence_sha256"] = _sha(evidence_path)
            receipt["outer_folds_sha256"] = _canonical(values["outer_folds"].tolist())
            evidence_receipt_path.write_bytes(_json_bytes(receipt))
            config["evidence_sha256"] = _sha(evidence_path)
            search_path = Path(config["v3_search_root"]) / "SEARCH_COMPLETE.json"
            search = json.loads(search_path.read_text()); search["evidence_sha256"] = config["evidence_sha256"]
            search_path.write_bytes(_json_bytes(search))
            calls = {"count": 0}
            config["fit_meta_callback"] = lambda *_: calls.__setitem__("count", calls["count"] + 1)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_search(config, Path(td) / "run")
            self.assertEqual(calls["count"], 0)

    def test_low_standalone_meta_does_not_skip_any_blend_or_later_head(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            config["fit_meta_callback"] = lambda declaration, p, y: {"declaration": declaration}
            config["predict_meta_callback"] = lambda artifact, p: np.tile([[0.9, 0.05, 0.05]], (p.shape[1], 1))
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_search(config, Path(td) / "run")
            rows = [json.loads(line) for line in Path(result["ledger_path"]).read_text().splitlines()]
            self.assertLess(rows[1]["standalone_meta_macro_f1"], rows[0]["macro_f1"])
            self.assertEqual(len(rows), 541)
            self.assertEqual(rows[-1]["declaration_index"], 53)
            self.assertEqual(rows[-1]["alpha"], 1.0)
            self.assertEqual(rows[-1]["pooling"], "geometric")

    def test_tampered_score_row_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            with redirect_stdout(io.StringIO()):
                run_search(config, Path(td) / "run")
            row_path = Path(td) / "run" / "scores" / "0540.json"
            row = json.loads(row_path.read_text()); row["macro_f1"] = 0.123
            row_path.write_bytes(_json_bytes(row))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "commit|immutable search artifact"):
                run_search(config, Path(td) / "run")

    def test_complete_resume_verifies_rows_without_reblending_or_rescoring(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            run_root = Path(td) / "run"
            with redirect_stdout(io.StringIO()):
                run_search(config, run_root)
            original_scores = meta_search_module._scores
            original_blend = meta_search_module.blend
            calls = {"scores": 0, "blend": 0}
            def counted_scores(*args, **kwargs):
                calls["scores"] += 1
                return original_scores(*args, **kwargs)
            def counted_blend(*args, **kwargs):
                calls["blend"] += 1
                return original_blend(*args, **kwargs)
            meta_search_module._scores = counted_scores
            meta_search_module.blend = counted_blend
            try:
                with redirect_stdout(io.StringIO()):
                    result = run_search(config, run_root)
            finally:
                meta_search_module._scores = original_scores
                meta_search_module.blend = original_blend
            self.assertEqual(result["declared_score_rows"], 541)
            self.assertEqual(calls, {"scores": 1, "blend": 0})

    def test_alpha_one_alias_semantics_are_validated_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            run_root = Path(td) / "run"
            with redirect_stdout(io.StringIO()):
                run_search(config, run_root)
            alias_path = run_root / "scores" / "0010.json"
            alias = json.loads(alias_path.read_text())
            alias["alias_of"] = 8
            alias.pop("row_sha256")
            alias["row_sha256"] = _canonical(alias)
            alias_payload = _json_bytes(alias)
            alias_path.write_bytes(alias_payload)
            commit_path = run_root / "score_commits" / "0010.json"
            commit = json.loads(commit_path.read_text())
            commit["row_bytes_sha256"] = hashlib.sha256(alias_payload).hexdigest()
            commit["row_sha256"] = alias["row_sha256"]
            commit_path.write_bytes(_json_bytes(commit))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "alias"):
                run_search(config, run_root)

    def test_complete_alternative_winner_resume_does_not_reblend_selection(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td), baseline_label_shift=1)
            run_root = Path(td) / "run"
            with redirect_stdout(io.StringIO()):
                first = run_search(config, run_root)
            self.assertTrue(first["deployment_required"])
            original_scores = meta_search_module._scores
            original_blend = meta_search_module.blend
            calls = {"scores": 0, "blend": 0}
            def counted_scores(*args, **kwargs):
                calls["scores"] += 1
                return original_scores(*args, **kwargs)
            def counted_blend(*args, **kwargs):
                calls["blend"] += 1
                return original_blend(*args, **kwargs)
            meta_search_module._scores = counted_scores
            meta_search_module.blend = counted_blend
            try:
                with redirect_stdout(io.StringIO()):
                    resumed = run_search(config, run_root)
            finally:
                meta_search_module._scores = original_scores
                meta_search_module.blend = original_blend
            self.assertTrue(resumed["deployment_required"])
            self.assertEqual(calls, {"scores": 1, "blend": 0})

    def test_partial_run_row_rewrite_with_new_self_checksum_is_rejected_by_commit(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            config["fit_meta_callback"] = lambda declaration, p, y: {"declaration": declaration}
            config["predict_meta_callback"] = lambda artifact, p: p[0]
            config["progress_callback"] = lambda event: (_ for _ in ()).throw(
                RuntimeError("stop after row one")
            ) if event["kind"] == "score" and event["declared_index"] == 1 else None
            run_root = Path(td) / "run"
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "row one"):
                run_search(config, run_root)
            row_path = run_root / "scores" / "0001.json"
            row = json.loads(row_path.read_text())
            row["macro_f1"] = 0.125
            row["per_class_f1"] = [0.125, 0.125, 0.125]
            row["probability_sha256"] = "a" * 64
            row.pop("row_sha256")
            row["row_sha256"] = _canonical(row)
            row_path.write_bytes(_json_bytes(row))
            config.pop("progress_callback")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "commit"):
                run_search(config, run_root)

    def test_existing_score_row_without_commit_fails_closed_and_is_retained(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            config["fit_meta_callback"] = lambda declaration, p, y: {"declaration": declaration}
            config["predict_meta_callback"] = lambda artifact, p: p[0]
            config["progress_callback"] = lambda event: (_ for _ in ()).throw(
                RuntimeError("stop after row one")
            ) if event["kind"] == "score" and event["declared_index"] == 1 else None
            run_root = Path(td) / "run"
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "row one"):
                run_search(config, run_root)
            commit_path = run_root / "score_commits" / "0001.json"
            self.assertTrue(commit_path.is_file(), "score row must have an independent durable commit")
            held_commit = run_root / "score_commits" / "0001.held"
            commit_path.rename(held_commit)
            row_path = run_root / "scores" / "0001.json"
            before = row_path.read_bytes()
            config.pop("progress_callback")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "commit missing"):
                run_search(config, run_root)
            self.assertEqual(row_path.read_bytes(), before)
            self.assertTrue(held_commit.is_file())

    def test_tampered_score_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            config["fit_meta_callback"] = lambda declaration, p, y: {"declaration": declaration}
            config["predict_meta_callback"] = lambda artifact, p: p[0]
            config["progress_callback"] = lambda event: (_ for _ in ()).throw(
                RuntimeError("stop after row one")
            ) if event["kind"] == "score" and event["declared_index"] == 1 else None
            run_root = Path(td) / "run"
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "row one"):
                run_search(config, run_root)
            commit_path = run_root / "score_commits" / "0001.json"
            self.assertTrue(commit_path.is_file(), "score row must have an independent durable commit")
            commit = json.loads(commit_path.read_text())
            commit["row_bytes_sha256"] = "0" * 64
            commit_path.write_bytes(_json_bytes(commit))
            config.pop("progress_callback")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "commit"):
                run_search(config, run_root)

    def test_interruption_after_commit_before_row_recomputes_only_missing_row_and_matches_commit(self):
        with tempfile.TemporaryDirectory() as td:
            config = _baseline_fixture(td, _bank_fixture(td))
            config["fit_meta_callback"] = lambda declaration, p, y: {"declaration": declaration}
            config["predict_meta_callback"] = lambda artifact, p: p[0]
            run_root = Path(td) / "run"
            target_row = run_root / "scores" / "0001.json"
            original_write_once = meta_search_module._write_once
            interrupted = {"value": False}
            def interrupt_row_publish(path, payload):
                if Path(path) == target_row and not interrupted["value"]:
                    interrupted["value"] = True
                    raise RuntimeError("interrupt after commit")
                return original_write_once(path, payload)
            meta_search_module._write_once = interrupt_row_publish
            try:
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "after commit"):
                    run_search(config, run_root)
            finally:
                meta_search_module._write_once = original_write_once
            commit_path = run_root / "score_commits" / "0001.json"
            self.assertTrue(commit_path.is_file(), "commit must publish before its score row")
            self.assertFalse(target_row.exists())
            commit_before = commit_path.read_bytes()
            with redirect_stdout(io.StringIO()):
                result = run_search(config, run_root)
            self.assertEqual(result["declared_score_rows"], 541)
            self.assertEqual(commit_path.read_bytes(), commit_before)
            self.assertTrue(target_row.is_file())


if __name__ == "__main__":
    unittest.main()
