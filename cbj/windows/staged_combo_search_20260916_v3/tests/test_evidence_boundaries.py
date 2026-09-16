from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evidence import (
    _copy_or_verify,
    _save_or_verify_npz,
    _verify_stage_reproduction,
)


class EvidenceBoundaryTests(unittest.TestCase):
    def test_stage_reproduction_accepts_tiny_float_drift_with_identical_argmax(self):
        expected = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
        reproduced = expected.copy()
        reproduced[0, 0] += 3e-16
        receipt = _verify_stage_reproduction(reproduced, expected)
        self.assertLessEqual(receipt["max_abs"], 1e-12)
        self.assertTrue(receipt["argmax_identical"])
        self.assertNotEqual(receipt["reproduced_probability_sha256"], receipt["stored_probability_sha256"])
        changed = expected.copy(); changed[0] = [0.1, 0.9]
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            _verify_stage_reproduction(changed, expected)

    def test_partial_evidence_artifacts_recover_missing_files_without_overwrite(self):
        arrays = {"ids": np.asarray(["R0", "R1"], dtype="U"), "p": np.asarray([[0.7, 0.3], [0.2, 0.8]])}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "EVIDENCE.npz"
            _save_or_verify_npz(npz, arrays)
            original_npz = npz.read_bytes()
            source = root / "source.json"; source.write_text(json.dumps({"x": 1}))
            copy = root / "stage_deployment.json"
            _copy_or_verify(copy, source)
            copy.unlink()
            _save_or_verify_npz(npz, arrays)
            _copy_or_verify(copy, source)
            self.assertEqual(npz.read_bytes(), original_npz)
            self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(), hashlib.sha256(source.read_bytes()).hexdigest())
            with npz.open("wb") as stream:
                np.savez_compressed(stream, ids=np.asarray(["OTHER"], dtype="U"), p=np.asarray([[0.5, 0.5]]))
            with self.assertRaisesRegex(ValueError, "changed"):
                _save_or_verify_npz(npz, arrays)


if __name__ == "__main__": unittest.main()
