import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tests.catboost_double import FakeCatBoostClassifier, patched_backend


class SyntheticGpuSmokeTests(unittest.TestCase):
    def test_smoke_runs_two_sequential_26_class_roundtrip_fits(self):
        from synthetic_gpu_smoke import run_smoke

        with tempfile.TemporaryDirectory() as temporary, patched_backend():
            receipt = run_smoke(temporary)

        self.assertEqual(receipt["status"], "SYNTHETIC_GPU_SMOKE_COMPLETE")
        self.assertEqual(receipt["classes"], 26)
        self.assertEqual(receipt["fit_count"], 2)
        self.assertEqual(receipt["iterations"], 5)
        self.assertEqual(receipt["depth"], 3)
        self.assertEqual(FakeCatBoostClassifier.fit_calls, 2)
        self.assertLessEqual(receipt["reload_max_abs_diff"], 1e-12)


if __name__ == "__main__":
    unittest.main()
