from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eligibility_audit import audit_admission, audit_receipts
from source_closure import build_source_closure, declared_payload_files, verify_source_closure


class SourceClosureTests(unittest.TestCase):
    def test_admission_has_128_declared_124_executable_and_exact_four_gpu_blocked(self):
        report = audit_admission(ROOT)
        self.assertEqual(report["declared_count"], 128)
        self.assertEqual(report["counts"], {"EXECUTABLE": 124, "BLOCKED": 4, "INELIGIBLE": 0})
        blocked = [row for row in report["declarations"] if row["status"] == "BLOCKED"]
        self.assertTrue(all(row["family"] == "T4" and row["parameters"]["head"] == "LightGBM_fixed" for row in blocked))
        self.assertTrue(all("GPU_BUILD_UNAVAILABLE" in row["reasons"] for row in blocked))

    def test_acceptance_receipts_are_present_and_passing(self):
        result = audit_receipts(ROOT)
        self.assertTrue(result["all_pass"])
        self.assertGreaterEqual(len(result["receipts"]), 10)

    def test_source_closure_verifies_after_relocation_and_rejects_change_or_rogue_source(self):
        allowlist = declared_payload_files(ROOT)
        closure = build_source_closure(ROOT, allowlist)
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory) / "bundle"
            relocated.mkdir()
            for relative in allowlist:
                target = relocated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            verify_source_closure(relocated, closure)
            (relocated / allowlist[0]).write_bytes((relocated / allowlist[0]).read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_source_closure(relocated, closure)
        rogue = ROOT / "rogue_untracked_source.py"
        try:
            rogue.write_text("raise RuntimeError\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "undeclared"):
                declared_payload_files(ROOT)
        finally:
            rogue.unlink()


if __name__ == "__main__":
    unittest.main()
