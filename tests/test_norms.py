# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.common import ZeroCenteredRMSNorm


class TestZeroCenteredRMSNorm(unittest.TestCase):
    def test_matches_rms_norm_with_one_plus_weight(self):
        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype):
                mx.random.seed(0)
                norm = ZeroCenteredRMSNorm(16, eps=1e-6)
                norm.weight = (0.1 * mx.random.normal((16,))).astype(dtype)
                reference = nn.RMSNorm(16, eps=1e-6)
                reference.weight = 1.0 + norm.weight
                x = mx.random.normal((2, 3, 16)).astype(dtype)
                self.assertTrue(mx.array_equal(norm(x), reference(x)))

    def test_starts_with_scale_one(self):
        norm = ZeroCenteredRMSNorm(8, eps=1e-5)
        x = mx.random.normal((4, 8))
        expected = mx.fast.rms_norm(x, mx.ones((8,)), 1e-5)
        self.assertTrue(mx.array_equal(norm(x), expected))


if __name__ == "__main__":
    unittest.main()
