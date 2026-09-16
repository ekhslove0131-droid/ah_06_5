from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifyReleaseCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.release = Path(self.temporary.name) / "cbj"
        package = self.release / "windows" / "pkg"
        package.mkdir(parents=True)
        (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (package / "SOURCE_ALLOWLIST.txt").write_text(
            "SOURCE_ALLOWLIST.txt\nmodule.py\n", encoding="utf-8"
        )
        source_rows = [
            {"path": "SOURCE_ALLOWLIST.txt", "sha256": sha256(package / "SOURCE_ALLOWLIST.txt")},
            {"path": "module.py", "sha256": sha256(package / "module.py")},
        ]
        source_identity = hashlib.sha256(
            (json.dumps(source_rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        (package / "SOURCE_MANIFEST.json").write_text(
            json.dumps({"source_sha256": source_identity, "files": source_rows}) + "\n",
            encoding="utf-8",
        )
        (self.release / "PRIVATE_INPUTS.json").write_text(
            json.dumps(
                {
                    "schema_version": "CBJ_PRIVATE_INPUTS_V1",
                    "inputs": [
                        {
                            "name": "TRAINING_BANK.npz",
                            "role": "frozen probability bank",
                            "sha256": "a" * 64,
                            "shape": [7, 6201, 26],
                            "included": False,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    @property
    def script(self) -> Path:
        return Path(__file__).resolve().parents[1] / "verify_release.py"

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.assertTrue(self.script.is_file(), "verify_release.py is required")
        return subprocess.run(
            [sys.executable, "-B", str(self.script), "--root", str(self.release), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def write_manifest(self) -> None:
        result = self.run_cli("--write-manifest")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_clean_release_reports_private_inputs_as_not_included(self) -> None:
        self.write_manifest()

        result = self.run_cli()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
        self.assertIn("TRAINING_BANK.npz: NOT_INCLUDED", result.stdout)

    def test_corrupted_tracked_file_fails_hash_verification(self) -> None:
        self.write_manifest()
        module = self.release / "windows" / "pkg" / "module.py"
        module.write_text("VALUE = 2\n", encoding="utf-8")

        result = self.run_cli()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash mismatch", result.stderr)

    def test_forbidden_csv_is_rejected_even_after_manifest_regeneration(self) -> None:
        (self.release / "leaked_rows.csv").write_text("ID,SUBCLASS\nA,BRCA\n", encoding="utf-8")

        result = self.run_cli("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden artifact", result.stderr)

    def test_notebook_with_output_is_rejected(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": ["secret\n"]}],
                    "source": ["print('secret')"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (self.release / "unsafe.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

        result = self.run_cli("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("notebook contains outputs", result.stderr)

    def test_python_syntax_error_is_rejected(self) -> None:
        package = self.release / "windows" / "pkg"
        (package / "module.py").write_text("def broken(:\n", encoding="utf-8")

        result = self.run_cli("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python syntax error", result.stderr)

    def test_recovery_package_verifies_own_identity_not_historical_dependency_closure(self) -> None:
        recovery = self.release / "windows" / "recovery"
        artifacts = recovery / "artifacts"
        artifacts.mkdir(parents=True)
        (recovery / "grade_runner.py").write_text("VALUE = 1\n", encoding="utf-8")
        (recovery / "runtime_config.recovery.windows.json").write_text("{}\n", encoding="utf-8")
        rows = [
            {"path": "grade_runner.py", "sha256": sha256(recovery / "grade_runner.py")},
            {
                "path": "runtime_config.recovery.windows.json",
                "sha256": sha256(recovery / "runtime_config.recovery.windows.json"),
            },
        ]
        identity = hashlib.sha256(
            (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        (artifacts / "RECOVERY_ACCEPTANCE_20260916.json").write_text(
            json.dumps({"source_sha256": identity}) + "\n", encoding="utf-8"
        )
        (artifacts / "SOURCE_CLOSURE.json").write_text(
            json.dumps(
                {
                    "schema_version": "TEAM_SOURCE_CLOSURE_V1",
                    "files": {"grade_runner.py": "0" * 64},
                    "dependency_role": "historical predecessor",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = self.run_cli("--write-manifest")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
