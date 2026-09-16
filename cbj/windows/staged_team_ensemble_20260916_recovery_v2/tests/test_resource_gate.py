from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource_gate import ResourceGateError, admit_resources


class ResourceGateTests(unittest.TestCase):
    def test_gpu_candidate_requires_cuda_and_free_memory(self):
        with self.assertRaisesRegex(ResourceGateError, "CUDA"):
            admit_resources("GPU_REQUIRED", 20_000, 1_000, None, 4_000)
        with self.assertRaisesRegex(ResourceGateError, "GPU memory"):
            admit_resources("GPU_REQUIRED", 20_000, 1_000, 3_999, 4_000)
        receipt = admit_resources("GPU_REQUIRED", 20_000, 1_000, 6_000, 4_000)
        self.assertEqual(receipt["device"], "cuda:0")
        self.assertFalse(receipt["cpu_fallback"])

    def test_host_memory_and_intentional_cpu_are_distinct(self):
        with self.assertRaisesRegex(ResourceGateError, "host memory"):
            admit_resources("CPU_INTENTIONAL", 999, 1_000, None, 0)
        receipt = admit_resources("CPU_INTENTIONAL", 2_000, 1_000, None, 0)
        self.assertEqual(receipt["device"], "cpu")
        self.assertEqual(receipt["threads"], 4)


if __name__ == "__main__":
    unittest.main()
