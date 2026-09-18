import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import crossblend


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _probability_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _identity():
    return {
        "ids": np.asarray(["I0", "I1", "I2", "I3"]),
        "y": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "groups": np.asarray(["G0", "G1", "G2", "G3"]),
        "inner_folds": np.asarray([0, 1, 2, 0], dtype=np.int16),
        "outer_folds": np.asarray([1, 2, 3, 4], dtype=np.int16),
        "class_order": np.asarray(["A", "B"]),
    }


def _write_inner_export(root, role, probability, *, parent_source=None, parent_config=None):
    root = Path(root)
    path = root / f"{role}.npz"
    arrays = {**_identity(), "probability": np.asarray(probability, dtype=np.float64)}
    np.savez_compressed(path, **arrays)
    receipt = {
        "schema_version": "STACK7_CROSSBLEND_INNER_EXPORT_V1",
        "status": "COMPLETE",
        "role": role,
        "own_source_sha256": "a" * 64,
        "own_runtime_config_sha256": "b" * 64,
        "parent_source_sha256": parent_source or "c" * 64,
        "parent_runtime_config_sha256": parent_config or "d" * 64,
        "parent_selection_sha256": "e" * 64,
        "parent_search_complete_sha256": "f" * 64,
        "selected_inner_probability_sha256": _probability_sha(probability),
        "npz_sha256": _sha(path),
        "test_read": False,
        "full_bank_read": False,
        "fit_count": 0,
    }
    receipt_path = root / f"{role}.json"
    receipt_path.write_bytes(_json_bytes(receipt))
    return path, receipt_path


def _parents():
    return {
        "linear": {
            "source_root": "/linear", "runtime_config_path": "/linear/config.json",
            "expected_source_sha256": "c" * 64, "expected_runtime_config_sha256": "d" * 64,
        },
        "nonlinear": {
            "source_root": "/nonlinear", "runtime_config_path": "/nonlinear/config.json",
            "expected_source_sha256": "c" * 64, "expected_runtime_config_sha256": "d" * 64,
        },
    }


def _current_inner_worker(probabilities):
    def worker(_config, role, mode, output_root):
        if mode != "inner":
            raise AssertionError("test worker permits inner verification only")
        return _write_inner_export(output_root, role, probabilities[role])
    return worker


class CrossblendTests(unittest.TestCase):
    def test_blend_has_literal_arithmetic_geometric_and_exact_endpoints(self):
        linear = np.asarray([[0.8, 0.2], [0.25, 0.75]])
        nonlinear = np.asarray([[0.2, 0.8], [0.75, 0.25]])
        np.testing.assert_array_equal(crossblend.blend_probability(linear, nonlinear, 0.0, "geometric"), linear)
        np.testing.assert_array_equal(crossblend.blend_probability(linear, nonlinear, 1.0, "arithmetic"), nonlinear)
        np.testing.assert_allclose(
            crossblend.blend_probability(linear, nonlinear, 0.25, "arithmetic")[0],
            [0.65, 0.35], rtol=0.0, atol=1e-15,
        )
        geometric = crossblend.blend_probability(linear[:1], nonlinear[:1], 0.5, "geometric")
        np.testing.assert_allclose(geometric[0], [0.5, 0.5], rtol=0.0, atol=1e-15)

    def test_search_enumerates_202_rows_with_two_endpoint_aliases_and_incumbent_first_ties(self):
        y = np.asarray([0, 1], dtype=np.int64)
        same = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
        rows, selection = crossblend.search_probabilities(y, same, same)
        self.assertEqual(len(rows), 202)
        self.assertEqual(sum(row["alias_of"] is not None for row in rows), 2)
        self.assertEqual(rows[0]["alpha"], 0.0)
        self.assertEqual(rows[0]["pooling"], "arithmetic")
        self.assertEqual(selection["declared_index"], 0)
        self.assertEqual(selection["alpha"], 0.0)

    def test_inner_exports_require_exact_identity_and_probability_member(self):
        linear = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            linear_path, linear_receipt = _write_inner_export(root, "linear", linear)
            nonlinear_path, nonlinear_receipt = _write_inner_export(root, "nonlinear", linear)
            loaded_linear = crossblend.load_inner_export(linear_path, linear_receipt, "linear")
            loaded_nonlinear = crossblend.load_inner_export(nonlinear_path, nonlinear_receipt, "nonlinear")
            loaded_nonlinear["groups"][0] = "DIFFERENT"
            with self.assertRaisesRegex(ValueError, "identity"):
                crossblend.validate_same_inner_identity(loaded_linear, loaded_nonlinear)

            bad = root / "bad.npz"
            np.savez_compressed(bad, **_identity(), valid_probability=linear)
            receipt = json.loads(nonlinear_receipt.read_text())
            receipt["npz_sha256"] = _sha(bad)
            bad_receipt = root / "bad.json"; bad_receipt.write_bytes(_json_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "probability"):
                crossblend.load_inner_export(bad, bad_receipt, "nonlinear")

    def test_parent_or_export_tamper_is_rejected(self):
        probability = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        with tempfile.TemporaryDirectory() as temporary:
            path, receipt_path = _write_inner_export(temporary, "linear", probability)
            receipt = json.loads(receipt_path.read_text())
            receipt["parent_selection_sha256"] = "x" * 64
            receipt_path.write_bytes(_json_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "parent"):
                crossblend.load_inner_export(path, receipt_path, "linear")

    def test_search_reads_only_inner_exports_and_reuses_identical_complete_outputs(self):
        linear = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        nonlinear = np.asarray([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5], [0.2, 0.8]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lp, lr = _write_inner_export(root, "linear", linear)
            np_, nr = _write_inner_export(root, "nonlinear", nonlinear)
            config = {
                "own_source_sha256": "a" * 64, "_runtime_config_sha256": "b" * 64,
                "parents": _parents(),
                "linear_inner_npz": str(lp), "linear_inner_receipt_json": str(lr),
                "nonlinear_inner_npz": str(np_), "nonlinear_inner_receipt_json": str(nr),
                "parent_deployment_poison": str(root / "MUST_NOT_READ"),
            }
            run = root / "search"
            worker = _current_inner_worker({"linear": linear, "nonlinear": nonlinear})
            first = crossblend.run_search(config, run, worker=worker)
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            second = crossblend.run_search(config, run, worker=worker)
            after = {path.name: path.read_bytes() for path in run.iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            self.assertEqual(first["declared_rows"], 202)
            self.assertFalse(first["full_read"])
            self.assertFalse(first["test_read"])
            self.assertEqual(first["fit_count"], 0)

            changed = dict(config); changed["_runtime_config_sha256"] = "z" * 64
            with self.assertRaisesRegex(ValueError, "binding|immutable"):
                crossblend.run_search(changed, run, worker=worker)

    def test_csv_maps_exact_ids_to_class_order_argmax(self):
        payload = crossblend.submission_csv_bytes(
            np.asarray(["T2", "T1"]),
            np.asarray([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]),
            np.asarray(["A", "B", "C"]),
        )
        rows = list(csv.reader(payload.decode().splitlines()))
        self.assertEqual(rows, [["ID", "SUBCLASS"], ["T2", "B"], ["T1", "A"]])

    def test_deployment_rejects_parent_identity_mismatch_before_publication(self):
        reference = {
            "ids": np.asarray(["I0", "I1"]), "y": np.asarray([0, 1]),
            "groups": np.asarray(["G0", "G1"]), "folds": np.asarray([0, 1]),
            "class_order": np.asarray(["A", "B"]), "test_ids": np.asarray(["T0"]),
            "oof_probability": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
            "test_probability": np.asarray([[0.7, 0.3]]),
        }
        other = {key: value.copy() for key, value in reference.items()}
        other["ids"][0] = "BAD"
        with self.assertRaisesRegex(ValueError, "identity"):
            crossblend.compose_deployment(reference, other, 0.5, "arithmetic")

    def test_completed_deployment_reuses_exact_outputs_and_rejects_receipt_or_csv_change(self):
        arrays = {
            "ids": np.asarray(["I0", "I1"]), "y": np.asarray([0, 1]),
            "groups": np.asarray(["G0", "G1"]), "folds": np.asarray([0, 1]),
            "class_order": np.asarray(["A", "B"]), "test_ids": np.asarray(["T0"]),
            "oof_probability": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
            "test_probability": np.asarray([[0.7, 0.3]]),
        }
        receipt = {"schema_version": "TEST", "status": "COMPLETE", "fit_count": 0}
        with tempfile.TemporaryDirectory() as temporary:
            first = crossblend._publish_deployment(temporary, arrays, receipt)
            second = crossblend._publish_deployment(temporary, arrays, receipt)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "immutable"):
                crossblend._publish_deployment(temporary, arrays, {**receipt, "status": "CHANGED"})
            Path(temporary, "submission_stack7_crossblend_frozen.csv").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "immutable"):
                crossblend._publish_deployment(temporary, arrays, receipt)

    def test_deployment_rejects_unbound_partial_files_without_overwriting_them(self):
        arrays = {
            "ids": np.asarray(["I0", "I1"]), "y": np.asarray([0, 1]),
            "groups": np.asarray(["G0", "G1"]), "folds": np.asarray([0, 1]),
            "class_order": np.asarray(["A", "B"]), "test_ids": np.asarray(["T0"]),
            "oof_probability": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
            "test_probability": np.asarray([[0.7, 0.3]]),
        }
        receipt = {"schema_version": "TEST", "status": "COMPLETE", "fit_count": 0}
        for name in (".predictions.npz.partial", ".submission.csv.partial"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                partial = Path(temporary, name); partial.write_bytes(b"foreign-partial")
                before = partial.read_bytes()
                with self.assertRaisesRegex(ValueError, "unbound"):
                    crossblend._publish_deployment(temporary, arrays, receipt)
                self.assertEqual(partial.read_bytes(), before)

    def test_cached_export_is_reverified_against_current_parent_state(self):
        probability = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path, receipt_path = _write_inner_export(root, "linear", probability)
            config = {
                "own_source_sha256": "a" * 64, "_runtime_config_sha256": "b" * 64,
                "parents": _parents(), "linear_inner_npz": str(npz_path),
                "linear_inner_receipt_json": str(receipt_path),
            }
            called = {"count": 0}
            def current_parent(_config, role, mode, output_root):
                called["count"] += 1
                changed = probability.copy(); changed[0] = [0.75, 0.25]
                return _write_inner_export(output_root, role, changed)
            with self.assertRaisesRegex(ValueError, "current parent|probability|binding"):
                crossblend.export_inner(config, "linear", worker=current_parent)
            self.assertEqual(called["count"], 1)

    def test_cross_selection_rejects_self_consistent_selection_and_complete_rewrite(self):
        linear = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        nonlinear = np.asarray([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5], [0.2, 0.8]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lp, lr = _write_inner_export(root, "linear", linear)
            np_, nr = _write_inner_export(root, "nonlinear", nonlinear)
            run = root / "search"
            config = {
                "own_source_sha256": "a" * 64, "_runtime_config_sha256": "b" * 64,
                "parents": _parents(),
                "linear_inner_npz": str(lp), "linear_inner_receipt_json": str(lr),
                "nonlinear_inner_npz": str(np_), "nonlinear_inner_receipt_json": str(nr),
                "cross_search_run_root": str(run),
            }
            worker = _current_inner_worker({"linear": linear, "nonlinear": nonlinear})
            crossblend.run_search(config, run, worker=worker)
            selection_path = run / "FROZEN_SELECTION.json"
            selection = json.loads(selection_path.read_text())
            selection["alpha"] = 0.99
            selection_path.write_bytes(_json_bytes(selection))
            complete_path = run / "SEARCH_COMPLETE.json"
            complete = json.loads(complete_path.read_text())
            complete["selection_sha256"] = _sha(selection_path)
            complete_path.write_bytes(_json_bytes(complete))
            with self.assertRaisesRegex(ValueError, "winner|ledger|selection"):
                crossblend._verify_cross_selection(config, worker=worker)

    def test_completed_run_deploy_reuses_verified_outputs_without_full_parent_workers(self):
        linear = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        nonlinear = np.asarray([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5], [0.2, 0.8]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lp, lr = _write_inner_export(root, "linear", linear)
            np_, nr = _write_inner_export(root, "nonlinear", nonlinear)
            search = root / "search"; deployment = root / "deployment"
            config = {
                "own_source_sha256": "a" * 64, "_runtime_config_sha256": "b" * 64,
                "parents": _parents(),
                "linear_inner_npz": str(lp), "linear_inner_receipt_json": str(lr),
                "nonlinear_inner_npz": str(np_), "nonlinear_inner_receipt_json": str(nr),
                "cross_search_run_root": str(search),
                "canonical_full_npz_sha256": "1" * 64,
                "sample_submission_sha256": "2" * 64,
            }
            inner_worker = _current_inner_worker({"linear": linear, "nonlinear": nonlinear})
            crossblend.run_search(config, search, worker=inner_worker)
            selection, binding = crossblend._verify_cross_selection(config, worker=inner_worker)
            arrays = {
                "ids": np.asarray(["I0", "I1"]), "y": np.asarray([0, 1]),
                "groups": np.asarray(["G0", "G1"]), "folds": np.asarray([0, 1]),
                "class_order": np.asarray(["A", "B"]), "test_ids": np.asarray(["T0"]),
                "oof_probability": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
                "test_probability": np.asarray([[0.7, 0.3]]),
            }
            receipt = {
                "schema_version": "STACK7_CROSSBLEND_DEPLOYMENT_COMPLETE_V1", "status": "COMPLETE",
                "fit_count": 0, "selected_alpha": selection["alpha"],
                "selected_pooling": selection["pooling"], "own_source_sha256": "a" * 64,
                "runtime_config_sha256": "b" * 64, "search_binding": binding,
                "canonical_full_npz_sha256": "1" * 64, "sample_submission_sha256": "2" * 64,
            }
            expected = crossblend._publish_deployment(deployment, arrays, receipt)
            def worker(_config, role, mode, output_root):
                if mode == "deployment":
                    raise AssertionError("completed deployment loaded full parent outputs")
                return inner_worker(_config, role, mode, output_root)
            self.assertEqual(crossblend.run_deploy(config, deployment, worker=worker), expected)

    def test_cli_requires_start_for_all_writing_phases(self):
        for command in ("export-inner", "search", "deploy"):
            with self.subTest(command=command):
                self.assertEqual(crossblend.main([command, "--config", "/missing.json"]), 2)


if __name__ == "__main__":
    unittest.main()
