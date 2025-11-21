# ABOUTME: Validates BatchKVCache handles batch-size shrink between updates.
# ABOUTME: Reproduces a regression where decode batches smaller than prefill crash.

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BatchKVCacheBatchMismatchTest(unittest.TestCase):
    def test_update_and_fetch_handles_smaller_batch(self):
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, '{ROOT}')
            import mlx.core as mx
            from mlx_lm.models.cache import BatchKVCache

            cache = BatchKVCache(left_padding=[0, 0, 0, 0])
            k0 = mx.zeros((4, 2, 1, 3))
            v0 = mx.zeros_like(k0)
            cache.update_and_fetch(k0, v0)

            k1 = mx.zeros((2, 2, 1, 3))
            v1 = mx.zeros_like(k1)
            cache.update_and_fetch(k1, v1)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
