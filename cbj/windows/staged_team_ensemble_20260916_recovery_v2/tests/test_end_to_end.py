from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_state import read_json
from runner import InjectedExit
from workflow import run_synthetic_acceptance


def manifest():
    return {
        "source_sha256": "a" * 64, "runtime_sha256": "b" * 64,
        "data_sha256": "c" * 64, "partition_sha256": "d" * 64,
        "admission_sha256": "e" * 64, "class_order_sha256": "f" * 64,
    }


class EndToEndTests(unittest.TestCase):
    def test_complete_synthetic_bundle_resumes_and_never_reads_forbidden_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            train = base / "train_fixture.txt"; train.write_text("synthetic training only\n", encoding="utf-8")
            forbidden = [base / name for name in ("test.csv", "sample_submission.csv", "server_score.json")]
            for path in forbidden:
                path.write_text("FORBIDDEN_SENTINEL\n", encoding="utf-8")
            accessed = []
            original = Path.read_text
            def tracked(path, *args, **kwargs):
                accessed.append(Path(path).resolve())
                return original(path, *args, **kwargs)
            run_root = base / "run"
            with patch.object(Path, "read_text", tracked):
                with self.assertRaises(InjectedExit):
                    run_synthetic_acceptance(
                        run_root, manifest(), train, fail_after_phase="round_manifest",
                    )
                result = run_synthetic_acceptance(run_root, manifest(), train)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed_phases"], [
                "feature_state", "model", "inner_oof", "round_manifest", "outer_refit", "fold_completion",
            ])
            self.assertTrue(all(path.resolve() not in accessed for path in forbidden))
            self.assertEqual(read_json(run_root / "synthetic/admission.json")["declared_count"], 128)
            round_receipt = result["receipts"]["round_manifest"]
            self.assertEqual(round_receipt["round0"], 280)
            self.assertTrue(round_receipt["meta_complete"])
            self.assertEqual(result["receipts"]["model"]["real_fit_calls"], 0)
            self.assertTrue(result["receipts"]["fold_completion"]["exact_outer_coverage"])
            self.assertTrue(all(not receipt.get("test_read", False) for receipt in result["receipts"].values()))


if __name__ == "__main__":
    unittest.main()
