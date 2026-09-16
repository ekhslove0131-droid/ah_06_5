from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_state import read_json
from runner import PHASES, InjectedExit, launch_detached, run_resumable, status


def manifest(source="a" * 64):
    return {
        "source_sha256": source,
        "runtime_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
        "admission_sha256": "e" * 64,
        "class_order_sha256": "f" * 64,
    }


class RunnerTests(unittest.TestCase):
    def test_resume_skips_every_completed_boundary(self):
        for boundary in PHASES:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                calls = []
                callbacks = {name: (lambda current=name: calls.append(current) or {"phase": current}) for name in PHASES}
                with self.assertRaises(InjectedExit):
                    run_resumable(root, callbacks, manifest(), fail_after_phase=boundary)
                first_calls = list(calls)
                complete = run_resumable(root, callbacks, manifest())
                self.assertEqual(calls.count(boundary), 1)
                self.assertEqual(complete["status"], "COMPLETE")
                self.assertEqual(complete["completed_phases"], list(PHASES))
                self.assertEqual(calls[:len(first_calls)], first_calls)
                heartbeat = read_json(root / "HEARTBEAT.json")
                self.assertEqual(heartbeat["completion_counts"]["phases"], len(PHASES))
                self.assertEqual(heartbeat["status"], "COMPLETE")

    def test_source_change_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            callbacks = {name: (lambda current=name: {"phase": current}) for name in PHASES}
            with self.assertRaises(InjectedExit):
                run_resumable(root, callbacks, manifest(), fail_after_phase=PHASES[0])
            with self.assertRaisesRegex(ValueError, "source manifest"):
                run_resumable(root, callbacks, manifest("0" * 64))

    def test_detached_lock_rejects_second_live_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            def worker():
                time.sleep(0.8)
            first = launch_detached(root, worker, start=True)
            self.assertEqual(first["status"], "STARTED")
            second = launch_detached(root, worker, start=True)
            self.assertEqual(second["status"], "ALREADY_RUNNING")
            deadline = time.time() + 3
            while time.time() < deadline and status(root).get("runner_live"):
                time.sleep(0.05)

    def test_status_and_launch_without_start_do_not_create_run_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "never-created"
            self.assertEqual(status(root)["status"], "NOT_STARTED")
            self.assertEqual(launch_detached(root, lambda: None, start=False)["status"], "START_FLAG_REQUIRED")
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
