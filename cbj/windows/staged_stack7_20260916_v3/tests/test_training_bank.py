from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training_bank
import cli


def canonical(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()


def sequence(value): return canonical(np.asarray(value).tolist())
def probability_hash(value): return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()
def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value): Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.components = self.root / "components"; self.components.mkdir(parents=True)
        self.rows, self.classes = 10, 3
        self.ids = np.asarray([f"R{i}" for i in range(self.rows)], dtype="U")
        self.y = np.arange(self.rows) % self.classes
        self.labels = np.asarray([f"C{i}" for i in self.y], dtype="U")
        self.groups = np.asarray([f"G{i}" for i in range(self.rows)], dtype="U")
        self.folds = np.arange(self.rows) % 5
        self.class_order = ["C0", "C1", "C2"]
        outer = []
        for fold in range(5):
            valid = np.flatnonzero(self.folds == fold); train = np.flatnonzero(self.folds != fold)
            row = {"fold": fold, "train_indices": train.tolist(), "valid_indices": valid.tolist()}
            if fold == 0:
                inner_rows = train
                row["inner"] = []
                for inner_fold in range(3):
                    inner_valid = inner_rows[np.arange(len(inner_rows)) % 3 == inner_fold]
                    inner_train = np.asarray([value for value in inner_rows if value not in set(inner_valid)])
                    row["inner"].append({
                        "fold": inner_fold, "train_indices": inner_train.tolist(), "valid_indices": inner_valid.tolist(),
                    })
            outer.append(row)
        self.partitions = {"outer": outer}
        self.inner_indices = np.asarray(outer[0]["train_indices"])
        self.inner_folds = np.empty(len(self.inner_indices), dtype=np.int16)
        local = {int(value): index for index, value in enumerate(self.inner_indices)}
        for inner in outer[0]["inner"]:
            self.inner_folds[[local[int(value)] for value in inner["valid_indices"]]] = inner["fold"]
        rng = np.random.default_rng(4)
        self.candidate_ids = list(training_bank.MODEL_ORDER[2:])
        self.inner_candidates = np.stack([
            self.probability(rng, len(self.inner_indices)) for _ in self.candidate_ids
        ])
        self.anchor_calls = []
        self.declaration_hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in training_bank.MODEL_ORDER}
        self.base_source_sha = "b" * 64
        self.test_ids_hash = "t" * 64
        for model in training_bank.MODEL_ORDER:
            for outer_row in outer:
                self.write_component(model, outer_row)

    def probability(self, rng, rows):
        value = rng.random((rows, self.classes)); return value / value.sum(1, keepdims=True)

    def write_component(self, model, outer, *, binding_changes=None, corrupt_npz=False, include=True):
        fold = int(outer["fold"]); root = self.components / model / f"fold_{fold}"
        if not include: return
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash((model, fold))) % (2**32))
        valid = self.probability(rng, len(outer["valid_indices"]))
        path = root / "probability.npz"
        # Object test payload proves allow_pickle=False code never reads it.
        with path.open("wb") as stream: np.savez_compressed(stream, valid=valid, test=np.asarray([{"forbidden": True}], dtype=object))
        bindings = {
            "component": model, "declaration_sha256": self.declaration_hashes[model], "fold": fold,
            "source_sha256": self.base_source_sha,
            "train_ids_sha256": sequence(self.ids[outer["train_indices"]]),
            "train_y_sha256": sequence(self.labels[outer["train_indices"]]),
            "train_groups_sha256": sequence(self.groups[outer["train_indices"]]),
            "valid_ids_sha256": sequence(self.ids[outer["valid_indices"]]),
            "test_ids_sha256": self.test_ids_hash, "class_order": self.class_order,
        }
        bindings.update(binding_changes or {})
        write_json(root / "COMPLETE.json", {
            "schema_version": "GRADED_COMPONENT_CACHE_V1", "bindings": bindings,
            "npz_sha256": file_hash(path), "valid_probability_sha256": probability_hash(valid),
            "test_probability_sha256": "not-read",
        })
        if corrupt_npz: path.write_bytes(path.read_bytes() + b"tamper")

    def anchor_probability(self, model, inner, fit_indices, valid_indices, namespace, partition_sha):
        self.anchor_calls.append((model, inner["fold"], namespace, partition_sha))
        rng = np.random.default_rng(abs(hash((model, inner["fold"], "inner"))) % (2**32))
        return self.probability(rng, len(valid_indices))

    def context(self):
        return {
            "data": {
                "ids": self.ids, "y_int": self.y, "y_label": self.labels,
                "groups": self.groups, "folds": self.folds, "class_order": self.class_order,
            },
            "partitions": self.partitions,
            "reference": {
                "ids": self.ids, "y": self.y, "groups": self.groups,
                "folds": self.folds, "class_order": np.asarray(self.class_order, dtype="U"),
            },
            "evidence": {
                "ids": self.ids[self.inner_indices], "y": self.y[self.inner_indices],
                "groups": self.groups[self.inner_indices], "outer_folds": self.folds[self.inner_indices],
                "inner_folds": self.inner_folds, "class_order": np.asarray(self.class_order, dtype="U"),
                "candidate_ids": np.asarray(self.candidate_ids, dtype="U"),
                "candidate_probabilities": self.inner_candidates,
            },
            "anchor_probability": self.anchor_probability,
            "declaration_hashes": self.declaration_hashes,
            "base_source_sha256": self.base_source_sha,
            "component_root": self.components.parent,
            "component_subpath": "components",
            "test_ids_sha256": self.test_ids_hash,
            "sequence_sha256": sequence,
            "canonical_sha256": canonical,
        }


class TrainingBankTests(unittest.TestCase):
    def test_preflight_reports_exact_availability_blocker(self):
        with patch.object(training_bank, "_verified_context", side_effect=ValueError("missing component cache: X fold 2")):
            result = training_bank.preflight({})
        self.assertFalse(result["ready"])
        self.assertEqual(result["blocker"], "missing component cache: X fold 2")

    def test_cli_requires_explicit_start_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"; config.write_text("{}")
            self.assertEqual(cli.main(["prepare-bank", "--config", str(config)]), 2)

    def test_builds_exact_order_inner_and_full_train_only_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            bank, lineage = training_bank._assemble_bank(fixture.context())
            self.assertEqual(bank["model_ids"].tolist(), list(training_bank.MODEL_ORDER))
            self.assertEqual(bank["inner_probability"].shape, (7, 8, 3))
            self.assertEqual(bank["full_oof_probability"].shape, (7, 10, 3))
            np.testing.assert_array_equal(bank["inner_ids"], fixture.ids[fixture.folds != 0])
            np.testing.assert_array_equal(bank["full_ids"], fixture.ids)
            self.assertFalse(any("test" in key for key in bank))
            self.assertEqual(len(lineage["component_receipts"]), 35)
            self.assertEqual(len(fixture.anchor_calls), 6)
            self.assertEqual({call[2] for call in fixture.anchor_calls}, {
                "outer0-inner0-H1", "outer0-inner1-H1", "outer0-inner2-H1",
                "outer0-inner0-C10", "outer0-inner1-C10", "outer0-inner2-C10",
            })

    def test_component_tamper_swapped_model_class_row_and_fold_fail_closed(self):
        changes = (
            {"component": "OTHER"}, {"class_order": ["C1", "C0", "C2"]},
            {"valid_ids_sha256": "0" * 64}, {"fold": 4},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory); outer = fixture.partitions["outer"][0]
                fixture.write_component(training_bank.MODEL_ORDER[0], outer, binding_changes=change)
                with self.assertRaisesRegex(ValueError, "component.*binding"):
                    training_bank._assemble_bank(fixture.context())
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); outer = fixture.partitions["outer"][0]
            fixture.write_component(training_bank.MODEL_ORDER[0], outer, corrupt_npz=True)
            with self.assertRaisesRegex(ValueError, "component.*hash"):
                training_bank._assemble_bank(fixture.context())

    def test_missing_model_and_duplicate_fold_coverage_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            path = fixture.components / training_bank.MODEL_ORDER[-1] / "fold_4/COMPLETE.json"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing component"):
                training_bank._assemble_bank(fixture.context())
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); context = fixture.context()
            duplicate = dict(context["partitions"]["outer"][1])
            duplicate["fold"] = 5
            context["partitions"] = {"outer": context["partitions"]["outer"] + [duplicate]}
            with self.assertRaisesRegex(ValueError, "coverage"):
                training_bank._assemble_bank(context)

    def test_inner_reference_identity_and_candidate_order_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); context = fixture.context()
            context["evidence"]["ids"] = context["evidence"]["ids"][::-1]
            with self.assertRaisesRegex(ValueError, "inner.*identity"):
                training_bank._assemble_bank(context)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); context = fixture.context()
            context["evidence"]["candidate_ids"] = context["evidence"]["candidate_ids"].copy()
            context["evidence"]["candidate_ids"][0] = context["evidence"]["candidate_ids"][1]
            with self.assertRaisesRegex(ValueError, "candidate.*duplicated"):
                training_bank._assemble_bank(context)

    def test_full_group_crossing_outer_folds_fails_before_component_use(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); context = fixture.context()
            changed = fixture.groups.copy(); changed[1] = changed[0]
            context["data"]["groups"] = changed; context["reference"]["groups"] = changed
            with self.assertRaisesRegex(ValueError, "group crosses outer folds"):
                training_bank._assemble_bank(context)

    def test_swapped_inner_fold_labels_and_outer_partition_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); context = fixture.context()
            inner = context["evidence"]["inner_folds"].copy()
            inner[inner == 0] = 9; inner[inner == 1] = 0; inner[inner == 9] = 1
            context["evidence"]["inner_folds"] = inner
            with self.assertRaisesRegex(ValueError, "inner fold assignment"):
                training_bank._assemble_bank(context)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory); context = fixture.context()
            outer = context["partitions"]["outer"][1]
            outer["valid_indices"] = list(reversed(outer["valid_indices"]))
            with self.assertRaisesRegex(ValueError, "outer partition/reference fold"):
                training_bank._assemble_bank(context)

    def test_partial_pair_resume_recovers_match_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"; first = training_bank._save_bank(output, bank, lineage, {"config": "x"})
            receipt = output / "TRAINING_BANK.json"; receipt.unlink()
            second = training_bank._save_bank(output, bank, lineage, {"config": "x"})
            self.assertEqual(first, second)
            receipt.unlink()
            (output / "TRAINING_BANK.PENDING.json").unlink()
            with self.assertRaisesRegex(ValueError, "unbound training bank"):
                training_bank._save_bank(output, bank, lineage, {"config": "x"})

    def test_foreign_source_or_config_cannot_relabel_same_bank_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"
            training_bank._save_bank(output, bank, lineage, {"source": "a" * 64, "config": "b" * 64})
            with self.assertRaisesRegex(ValueError, "producer identity"):
                training_bank._save_bank(output, bank, lineage, {"source": "c" * 64, "config": "d" * 64})

    def test_unbound_tampered_bank_is_retained_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"; training_bank._save_bank(output, bank, lineage, {"config": "x"})
            receipt = output / "TRAINING_BANK.json"; receipt.unlink()
            (output / "TRAINING_BANK.PENDING.json").unlink()
            path = output / "TRAINING_BANK.npz"; path.write_bytes(path.read_bytes() + b"tamper")
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "unbound training bank"):
                training_bank._save_bank(output, bank, lineage, {"config": "x"})
            self.assertEqual(path.read_bytes(), original)

    def test_missing_readonly_anchor_cache_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            with self.assertRaisesRegex(ValueError, "anchor cache.*missing"):
                training_bank._readonly_anchor_artifact_map(
                    {"old_run_root": str(root / "absent")},
                    old_source_sha="1" * 64, old_parser_sha="2" * 64,
                    train_sha="3" * 64, make_cache_key=lambda bindings, supervised: "x",
                )
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            self.assertEqual(before, after)

    def test_coordinated_wrong_base_source_is_rejected_against_literal(self):
        wrong = "f" * 64
        with self.assertRaisesRegex(ValueError, "literal"):
            training_bank._require_base_source(wrong, wrong, wrong)

    def test_unbound_valid_npz_partial_is_retained_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"; output.mkdir()
            partial = output / ".TRAINING_BANK.npz.partial"
            with partial.open("wb") as stream: np.savez_compressed(stream, **bank)
            original = partial.read_bytes()
            with self.assertRaisesRegex(ValueError, "unbound partial"):
                training_bank._save_bank(output, bank, lineage, {"source": "a", "config": "b"})
            self.assertEqual(partial.read_bytes(), original)
            self.assertFalse((output / "TRAINING_BANK.npz").exists())

    def test_wrong_source_bound_npz_partial_is_retained_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"; identity = {"source": "a" * 64, "config": "b" * 64}
            training_bank._save_bank(output, bank, lineage, identity)
            (output / "TRAINING_BANK.json").unlink()
            final = output / "TRAINING_BANK.npz"; partial = output / ".TRAINING_BANK.npz.partial"
            final.rename(partial); original = partial.read_bytes()
            with self.assertRaisesRegex(ValueError, "producer identity"):
                training_bank._save_bank(output, bank, lineage, {"source": "c" * 64, "config": "d" * 64})
            self.assertEqual(partial.read_bytes(), original)

    def test_foreign_pending_partial_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"; output.mkdir()
            pending_partial = output / ".TRAINING_BANK.PENDING.json.partial"
            pending_partial.write_bytes(b"foreign-pending")
            original = pending_partial.read_bytes()
            with self.assertRaisesRegex(ValueError, "pending partial"):
                training_bank._save_bank(output, bank, lineage, {"source": "a", "config": "b"})
            self.assertEqual(pending_partial.read_bytes(), original)

    def test_matching_bound_npz_partial_resumes_successfully(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture = Fixture(root / "fixture")
            bank, lineage = training_bank._assemble_bank(fixture.context())
            output = root / "output"; identity = {"source": "a" * 64, "config": "b" * 64}
            expected = training_bank._save_bank(output, bank, lineage, identity)
            (output / "TRAINING_BANK.json").unlink()
            final = output / "TRAINING_BANK.npz"; partial = output / ".TRAINING_BANK.npz.partial"
            final.rename(partial)
            actual = training_bank._save_bank(output, bank, lineage, identity)
            self.assertEqual(expected, actual)
            self.assertTrue(final.is_file())
            self.assertFalse(partial.exists())


if __name__ == "__main__": unittest.main()
