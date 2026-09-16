from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_cli_has_launch_and_status_but_no_stop_or_kill(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("launch", help_text)
        self.assertIn("status", help_text)
        self.assertIn("preflight", help_text)
        self.assertNotIn(" stop", help_text)
        self.assertNotIn(" kill", help_text)

    def test_status_and_unarmed_launch_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            self.assertEqual(main(["status", "--run-root", str(root)]), 0)
            self.assertEqual(main(["launch", "--run-root", str(root)]), 2)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
