import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import cli
from artifact_store import sequence_sha256


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_bytes(_json_bytes(value))


def _grid_manifest():
    representations = ["probability", "log_probability", "log_probability_confidence"]
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    weightings = ["none", "sqrt_inverse", "balanced"]
    declarations = [
        {"representation": representation, "C": c_value, "weighting": weighting}
        for representation in representations
        for c_value in c_values
        for weighting in weightings
    ]
    return {
        "schema_version": "STACK7_META_GRID_V1",
        "declarations": declarations,
        "axes": {
            "representations": representations,
            "C": c_values,
            "weightings": weightings,
            "inner_folds": [0, 1, 2],
            "alphas": [0.1, 0.25, 0.5, 0.75, 1.0],
            "poolings": ["arithmetic", "geometric"],
        },
        "counts": {"heads": 54, "inner_fits": 162, "mixtures": 540, "controls": 1, "score_rows": 541},
        "ordering": "declarations_then_alpha_then_pooling_with_baseline_control_first",
        "alpha_one_geometric_is_alias": True,
    }


class CliTests(unittest.TestCase):
    def test_runtime_template_covers_all_persisted_task3_and_task4_api_keys(self):
        required = {
            "training_bank_npz", "training_bank_receipt_json", "training_bank_sha256",
            "training_bank_receipt_sha256", "bank_preparation_source_sha256",
            "bank_preparation_config_sha256", "expected_model_ids", "expected_inner_rows",
            "expected_full_rows", "expected_classes", "evidence_npz",
            "evidence_receipt_json", "evidence_sha256", "expected_baseline_inner_macro_f1",
            "v3_source_root", "v3_runtime_config_path", "v3_search_root",
            "v3_source_sha256", "v3_runtime_config_sha256", "v3_frozen_recipe_sha256",
            "meta_source_sha256", "meta_search_run_root", "meta_deployment_run_root",
            "v3_deployment_run_root", "v3_predictions_npz", "v3_predictions_sha256",
            "v3_deployment_complete_json", "v3_deployment_complete_sha256",
            "v3_submission_sha256", "expected_v3_full_oof_macro_f1",
            "sample_submission_csv", "sample_submission_sha256", "expected_test_rows",
            "v3_stage_deployment_json", "v3_stage_deployment_sha256",
        }
        future_derived = {"meta_search_selection_sha256", "meta_search_complete_sha256"}
        template = json.loads(Path(PACKAGE_ROOT, "runtime_config.template.json").read_text())
        windows = json.loads(Path(PACKAGE_ROOT, "runtime_config.windows.json").read_text())
        self.assertEqual(required - set(template), set())
        self.assertEqual(required - set(windows), set())
        self.assertEqual(future_derived & set(template), set())
        self.assertEqual(future_derived & set(windows), set())
        for key in ("schema_version", "expected_model_ids", "expected_inner_rows",
                    "expected_full_rows", "expected_classes", "expected_test_rows"):
            self.assertEqual(template[key], windows[key])

    def _package_fixture(self, root):
        root = Path(root)
        (root / "cli.py").write_text("# fixture\n")
        (root / "SOURCE_ALLOWLIST.txt").write_text("SOURCE_ALLOWLIST.txt\ncli.py\n")
        source_sha, rows = cli.source_identity(root)
        source_manifest = root / "SOURCE_MANIFEST.json"
        _write_json(source_manifest, {
            "schema_version": "STACK7_META_SOURCE_MANIFEST_V1",
            "source_sha256": source_sha,
            "files": rows,
        })
        grid = root / "GRID_MANIFEST.json"
        _write_json(grid, _grid_manifest())
        config_path = root / "runtime_config.windows.json"
        config = {
            "schema_version": "STACK7_META_RUNTIME_CONFIG_V1",
            "meta_source_sha256": source_sha,
            "source_manifest_json": str(source_manifest),
            "source_manifest_sha256": _sha(source_manifest),
            "grid_manifest_json": str(grid),
            "grid_manifest_sha256": _sha(grid),
            "prepared_source_json": str(root / "PREPARED_SOURCE.json"),
            "expected_model_ids": [f"M{i}" for i in range(7)],
            "expected_inner_rows": 6,
            "expected_full_rows": 10,
            "expected_classes": 2,
            "meta_search_run_root": str(root / "runs/search_meta_v1"),
            "meta_deployment_run_root": str(root / "runs/deployment_meta_v1"),
            "known_worker_run_roots": [str(root / "runs/search_meta_v1"), str(root / "runs/deployment_meta_v1")],
        }
        _write_json(config_path, config)
        prepared = {
            "schema_version": "STACK7_META_PREPARED_SOURCE_V1",
            "status": "CANDIDATE_LOCAL_SYNTHETIC_VERIFIED",
            "meta_source_sha256": source_sha,
            "source_files": rows,
            "source_file_count": len(rows),
            "source_manifest_sha256": _sha(source_manifest),
            "grid_manifest_sha256": _sha(grid),
            "runtime_config_windows_sha256": _sha(config_path),
            "declared_counts": {"heads": 54, "inner_fits": 162, "mixtures": 540, "controls": 1, "score_rows": 541},
        }
        _write_json(root / "PREPARED_SOURCE.json", prepared)
        return config_path, cli.load_runtime_config(config_path)

    def test_source_identity_rejects_duplicate_absolute_traversal_and_missing_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.py").write_text("pass\n")
            for content in (
                "source.py\nsource.py\n",
                "/tmp/source.py\n",
                "../source.py\n",
                "missing.py\n",
            ):
                (root / "SOURCE_ALLOWLIST.txt").write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    cli.source_identity(root)

    def test_package_verification_rejects_source_manifest_and_runtime_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, config = self._package_fixture(temporary)
            self.assertEqual(cli.verify_package(config, config_path, temporary)["status"], "PACKAGE_VERIFIED")
            Path(temporary, "cli.py").write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "source"):
                cli.verify_package(config, config_path, temporary)
        with tempfile.TemporaryDirectory() as temporary:
            config_path, config = self._package_fixture(temporary)
            Path(config_path).write_bytes(Path(config_path).read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "runtime config"):
                cli.verify_package(config, config_path, temporary)

    def test_grid_manifest_tamper_is_rejected_before_inputs_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path, config = self._package_fixture(temporary)
            manifest = json.loads(Path(config["grid_manifest_json"]).read_text())
            manifest["counts"]["score_rows"] = 540
            _write_json(config["grid_manifest_json"], manifest)
            with mock.patch.object(cli, "validate_search_inputs") as loader:
                with self.assertRaisesRegex(ValueError, "grid manifest"):
                    cli.preflight(config, config_path, temporary, phase="search")
                loader.assert_not_called()

    def test_search_input_preflight_reads_only_inner_bank_and_baseline(self):
        ids = np.asarray([f"I{i}" for i in range(6)])
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        groups = np.asarray([f"G{i}" for i in range(6)])
        outer = np.asarray([1, 2, 3, 4, 1, 2], dtype=np.int16)
        inner = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int16)
        probability = np.full((6, 2), 0.5, dtype=np.float64)
        bank = {
            "model_ids": np.asarray([f"M{i}" for i in range(7)]),
            "class_order": np.asarray(["A", "B"]),
            "inner_ids": ids, "inner_y": y, "inner_groups": groups,
            "inner_outer_folds": outer, "inner_folds": inner,
        }
        lineage = {
            "ids_sha256": sequence_sha256(ids), "y_sha256": sequence_sha256(y),
            "groups_sha256": sequence_sha256(groups), "outer_folds_sha256": sequence_sha256(outer),
            "inner_folds_sha256": sequence_sha256(inner),
            "class_order_sha256": sequence_sha256(np.asarray(["A", "B"])),
        }
        config = {"expected_model_ids": [f"M{i}" for i in range(7)]}
        with mock.patch.object(cli, "load_inner_bank", return_value=bank) as inner_load, \
                mock.patch.object(cli, "load_baseline_inner", return_value=(probability, np.zeros(6, dtype=bool), lineage)) as baseline_load, \
                mock.patch.object(cli, "load_full_bank", side_effect=AssertionError("full/test member loaded")) as full_load:
            result = cli.validate_search_inputs(config)
        self.assertEqual(result["status"], "INNER_INPUTS_VERIFIED")
        inner_load.assert_called_once_with(config)
        baseline_load.assert_called_once_with(config)
        full_load.assert_not_called()

    def test_search_and_deploy_without_start_do_not_create_run_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary, "runtime.json")
            _write_json(config_path, {})
            for command in ("search", "deploy"):
                with self.subTest(command=command):
                    code = cli.main([command, "--config", str(config_path)])
                    self.assertEqual(code, 2)
                    self.assertEqual(sorted(path.name for path in Path(temporary).iterdir()), ["runtime.json"])

    def test_duplicate_worker_blocks_start(self):
        config = {"known_worker_run_roots": ["/runs/a", "/runs/b"]}
        with self.assertRaisesRegex(ValueError, "active worker"):
            cli.ensure_no_active_workers(config, lambda path: str(path) == "/runs/b")

    def test_deployment_hash_resolution_is_in_memory_and_preserves_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            search = root / "search"; search.mkdir()
            selection = search / "FROZEN_SELECTION.json"; selection.write_text('{"status":"FROZEN"}\n')
            complete = search / "SEARCH_COMPLETE.json"; complete.write_text('{"status":"COMPLETE"}\n')
            config_path = root / "runtime.json"
            config = {"meta_search_run_root": str(search), "_runtime_config_sha256": "a" * 64}
            _write_json(config_path, {"meta_search_run_root": str(search)})
            before = config_path.read_bytes(); before_sha = _sha(config_path)
            resolved = cli.resolve_deployment_config(config)
            self.assertEqual(resolved["meta_search_selection_sha256"], _sha(selection))
            self.assertEqual(resolved["meta_search_complete_sha256"], _sha(complete))
            self.assertEqual(resolved["_runtime_config_sha256"], "a" * 64)
            self.assertNotIn("meta_search_selection_sha256", config)
            self.assertEqual(config_path.read_bytes(), before)
            self.assertEqual(_sha(config_path), before_sha)

    def test_incumbent_deployment_preflight_does_not_load_full_or_test(self):
        selection = {"status": "FROZEN", "deployment_required": False, "selected_kind": "baseline_control"}
        with mock.patch.object(cli, "verify_frozen_selection", return_value=(selection, {"status": "COMPLETE"})), \
                mock.patch.object(cli, "load_full_bank", side_effect=AssertionError("full bank loaded")) as full_load:
            result = cli.validate_deployment_freeze({"meta_search_run_root": "/fixed/search"})
        self.assertEqual(result["status"], "INCUMBENT_NO_DEPLOYMENT_READY")
        full_load.assert_not_called()

    def test_status_is_read_only_and_includes_package_and_inner_input_validation(self):
        class Runner:
            @staticmethod
            def _runner_live(_root):
                return False

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "runtime.json"; _write_json(config_path, {})
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            config = {"status_run_roots": {"meta_search": str(root / "not_started")}}
            with mock.patch.object(cli, "verify_package", return_value={"status": "PACKAGE_VERIFIED"}) as package_check, \
                    mock.patch.object(cli, "validate_search_inputs", return_value={"status": "INNER_INPUTS_VERIFIED"}) as input_check, \
                    mock.patch.object(cli, "load_verified_runner", return_value=Runner()):
                result = cli.status(config, config_path, root)
            after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
        self.assertEqual(result["package"]["status"], "PACKAGE_VERIFIED")
        self.assertEqual(result["inputs"]["status"], "INNER_INPUTS_VERIFIED")
        self.assertEqual(before, after)
        package_check.assert_called_once_with(config, config_path, root)
        input_check.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
