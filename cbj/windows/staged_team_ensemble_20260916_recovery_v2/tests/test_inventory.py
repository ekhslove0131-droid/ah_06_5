from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inventory import compile_admission, canonical_bytes


class InventoryTests(unittest.TestCase):
    def load_spaces(self):
        old = json.loads((ROOT / "fixtures/old_grid_80.json").read_text(encoding="utf-8"))
        team = json.loads((ROOT / "fixtures/team_grid_48.json").read_text(encoding="utf-8"))
        return old, team

    def test_exact_128_declarations_are_unique_and_accounted(self):
        old, team = self.load_spaces()
        report = compile_admission(old, team, support={})
        self.assertEqual(report["declared_count"], 128)
        self.assertEqual(report["source_counts"], {"OLD": 80, "TEAM": 48})
        self.assertEqual(sum(report["counts"].values()), 128)
        self.assertEqual(len(report["declarations"]), 128)
        self.assertEqual(len({row["config_hash"] for row in report["declarations"]}), 128)
        self.assertEqual(len({row["candidate_id"] for row in report["declarations"]}), 128)
        self.assertTrue(all(row["status"] == "BLOCKED" for row in report["declarations"]))
        self.assertTrue(all(row["reasons"] == ["ADAPTER_NOT_REGISTERED"] for row in report["declarations"]))

    def test_support_is_exact_config_hash_not_family_wide(self):
        old, team = self.load_spaces()
        blocked = compile_admission(old, team, support={})
        chosen = blocked["declarations"][0]
        support = {
            chosen["config_hash"]: {
                "status": "EXECUTABLE",
                "implementation_source_hash": "a" * 64,
                "device_kind": "CPU_INTENTIONAL",
            }
        }
        admitted = compile_admission(old, team, support=support)
        executable = [row for row in admitted["declarations"] if row["status"] == "EXECUTABLE"]
        self.assertEqual([row["config_hash"] for row in executable], [chosen["config_hash"]])
        self.assertEqual(executable[0]["reasons"], [])
        self.assertEqual(admitted["counts"], {"EXECUTABLE": 1, "BLOCKED": 127, "INELIGIBLE": 0})

    def test_output_is_byte_deterministic(self):
        old, team = self.load_spaces()
        first = canonical_bytes(compile_admission(old, team, support={}))
        second = canonical_bytes(compile_admission(old, team, support={}))
        self.assertEqual(first, second)

    def test_unapproved_status_or_unknown_support_hash_fails_closed(self):
        old, team = self.load_spaces()
        with self.assertRaisesRegex(ValueError, "support status"):
            compile_admission(old, team, support={"f" * 64: {"status": "READY"}})
        with self.assertRaisesRegex(ValueError, "unknown config"):
            compile_admission(old, team, support={"f" * 64: {"status": "EXECUTABLE"}})


if __name__ == "__main__":
    unittest.main()
