import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from artifact_store import (
    load_full_bank,
    load_inner_bank,
    load_or_fit_head,
)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _probability(rows=12, classes=3):
    value = np.full((7, rows, classes), 0.1, dtype=np.float64)
    for model in range(7):
        for row in range(rows):
            value[model, row, row % classes] = 0.8
    return value


def _bank_fixture(root, *, forbidden_full=False):
    root = Path(root)
    bank = root / "TRAINING_BANK.npz"
    receipt_path = root / "TRAINING_BANK.json"
    models = [f"M{i}" for i in range(7)]
    classes = ["A", "B", "C"]
    arrays = {
        "model_ids": np.asarray(models),
        "class_order": np.asarray(classes),
        "inner_ids": np.asarray([f"I{i}" for i in range(12)]),
        "inner_y": np.asarray([0, 1, 2] * 4, dtype=np.int64),
        "inner_groups": np.asarray([f"G{i}" for i in range(12)]),
        "inner_outer_folds": np.asarray([1, 2, 3, 4] * 3, dtype=np.int16),
        "inner_folds": np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int16),
        "inner_probability": _probability(),
    }
    if forbidden_full:
        forbidden = np.asarray([{"must": "not load"}], dtype=object)
        arrays.update({
            "full_ids": forbidden,
            "full_y": forbidden,
            "full_groups": forbidden,
            "full_outer_folds": forbidden,
            "full_oof_probability": forbidden,
        })
    else:
        arrays.update({
            "full_ids": np.asarray([f"F{i}" for i in range(15)]),
            "full_y": np.asarray([0, 1, 2] * 5, dtype=np.int64),
            "full_groups": np.asarray([f"FG{i}" for i in range(15)]),
            "full_outer_folds": np.asarray([0, 1, 2, 3, 4] * 3, dtype=np.int16),
            "full_oof_probability": _probability(15),
        })
    np.savez(bank, **arrays)
    identity = {"meta_source_sha256": "bank-source", "runtime_config_sha256": "bank-config"}
    receipt = {
        "schema_version": "STACK7_TRAINING_BANK_V2",
        "status": "COMPLETE",
        "bank_sha256": _sha(bank),
        "identity": identity,
        "array_sha256": {
            key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in arrays.items()
        },
        "keys": sorted(arrays),
        "model_order": models,
        "inner_shape": [7, 12, 3],
        "full_shape": [7, 15, 3],
        "test_probability_read": False,
        "test_data_read": False,
    }
    receipt_path.write_bytes(_json_bytes(receipt))
    config = {
        "training_bank_npz": str(bank),
        "training_bank_receipt_json": str(receipt_path),
        "training_bank_sha256": _sha(bank),
        "training_bank_receipt_sha256": _sha(receipt_path),
        "bank_preparation_source_sha256": "bank-source",
        "bank_preparation_config_sha256": "bank-config",
        "expected_model_ids": models,
        "expected_inner_rows": 12,
        "expected_full_rows": 15,
        "expected_classes": 3,
    }
    return config


def _identity():
    return {
        "source_sha256": "source",
        "runtime_config_sha256": "config",
        "bank_sha256": "bank",
        "recipe_sha256": "recipe",
        "declaration_sha256": "declaration",
        "fit_ids_sha256": "fit-ids",
        "fit_y_sha256": "fit-y",
        "fit_groups_sha256": "fit-groups",
        "fit_folds_sha256": "fit-folds",
        "valid_ids_sha256": "valid-ids",
        "valid_y_sha256": "valid-y",
        "valid_groups_sha256": "valid-groups",
        "valid_folds_sha256": "valid-folds",
        "model_order": [f"M{i}" for i in range(7)],
        "class_order": ["A", "B", "C"],
        "inner_fold": 0,
    }


def _deployment_identity(role="full_meta_outer_cv", fold=4):
    value = _identity(); value.pop("inner_fold")
    value.update({
        "schema_version": "STACK7_META_DEPLOYMENT_FIT_IDENTITY_V1",
        "feature_role": role, "fold": fold,
        "fit_probability_tensor_sha256": "a" * 64,
        "valid_probability_tensor_sha256": "b" * 64,
    })
    return value


class ArtifactStoreTests(unittest.TestCase):
    def test_deployment_identity_accepts_full_outer_folds_and_confirmation_zero(self):
        with tempfile.TemporaryDirectory() as td:
            calls = {"count": 0}
            def fit(): calls["count"] += 1; return {"constant": 1}
            predict = lambda _: np.tile([[0.7, 0.2, 0.1]], (4, 1))
            for identity in (_deployment_identity(fold=4), _deployment_identity("outer0_confirmation_inner_oof", 0)):
                load_or_fit_head(td, identity, fit, predict)
            self.assertEqual(calls["count"], 2)

    def test_deployment_identity_rejects_invalid_role_fold_hashes_and_extra_keys(self):
        invalid = []
        invalid.append({**_deployment_identity(fold=0), "feature_role": "unknown"})
        invalid.append({**_deployment_identity("outer0_confirmation_inner_oof", 1)})
        invalid.append({**_deployment_identity(fold=5)})
        invalid.append({**_deployment_identity(), "fit_probability_tensor_sha256": "bad"})
        invalid.append({**_deployment_identity(), "extra": True})
        for index, identity in enumerate(invalid):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
                load_or_fit_head(td, identity, lambda: None, lambda _: None)
    def test_inner_reader_never_loads_forbidden_full_object_members(self):
        with tempfile.TemporaryDirectory() as td:
            config = _bank_fixture(td, forbidden_full=True)
            bank = load_inner_bank(config)
            self.assertEqual(set(bank), {
                "model_ids", "class_order", "inner_ids", "inner_y", "inner_groups",
                "inner_outer_folds", "inner_folds", "inner_probability", "lineage",
            })
            self.assertEqual(bank["inner_probability"].shape, (7, 12, 3))
            with self.assertRaises(ValueError):
                load_full_bank(config)

    def test_bank_binding_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            config = _bank_fixture(td)
            for key, bad in (
                ("bank_preparation_source_sha256", "bad"),
                ("expected_model_ids", ["bad"] * 7),
                ("expected_classes", 4),
                ("expected_inner_rows", 11),
            ):
                changed = dict(config); changed[key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_inner_bank(changed)

    def test_fit_cache_reuses_verified_model_and_probability(self):
        with tempfile.TemporaryDirectory() as td:
            calls = {"fit": 0, "predict": 0}
            def fit():
                calls["fit"] += 1
                return {"constant": 1}
            def predict(model):
                calls["predict"] += 1
                self.assertEqual(model, {"constant": 1})
                return np.tile([[0.7, 0.2, 0.1]], (4, 1))

            first = load_or_fit_head(td, _identity(), fit, predict)
            second = load_or_fit_head(td, _identity(), fit, predict)

            self.assertEqual(calls, {"fit": 1, "predict": 1})
            self.assertEqual(first[2], second[2])
            np.testing.assert_array_equal(first[1], second[1])

    def test_tampered_model_is_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as td:
            result = load_or_fit_head(
                td, _identity(), lambda: {"constant": 1},
                lambda model: np.tile([[0.7, 0.2, 0.1]], (4, 1)),
            )
            model_path = Path(result[2]["model_path"])
            model_path.write_bytes(b"not a pickle")
            original = joblib.load
            joblib.load = lambda *_args, **_kwargs: self.fail("unpickle attempted before hash verification")
            try:
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_or_fit_head(td, _identity(), lambda: None, lambda model: None)
            finally:
                joblib.load = original

    def test_unbound_partial_is_retained_and_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / _canonical(_identity())
            target.mkdir()
            partial = target / ".model.joblib.partial"
            partial.write_bytes(b"foreign")
            with self.assertRaisesRegex(ValueError, "unbound partial"):
                load_or_fit_head(
                    td, _identity(), lambda: {"constant": 1},
                    lambda model: np.tile([[0.7, 0.2, 0.1]], (4, 1)),
                )
            self.assertEqual(partial.read_bytes(), b"foreign")


if __name__ == "__main__":
    unittest.main()
