# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models.switch_layers import SwitchGLU


def assert_allclose(testcase, left, right, rtol=1e-5, atol=1e-5):
    testcase.assertTrue(bool(mx.allclose(left, right, rtol=rtol, atol=atol)))


class TestSwitchGLUFusion(unittest.TestCase):
    def test_fused_gate_up_matches_unfused(self):
        mx.random.seed(0)
        layer = SwitchGLU(16, 32, 4, bias=True)
        layer.eval()
        x = mx.random.normal((8, 16))
        indices = mx.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]])

        layer.fuse_gate_up = False
        expected = layer(x, indices)
        layer.fuse_gate_up = True
        actual = layer(x, indices)
        mx.eval(expected, actual)

        assert_allclose(self, actual, expected)

    def test_fused_gate_up_matches_unfused_sorted_path(self):
        mx.random.seed(1)
        layer = SwitchGLU(16, 32, 4, bias=True)
        layer.eval()
        x = mx.random.normal((64, 16))
        indices = mx.array([[i % 4, (i + 1) % 4] for i in range(64)])

        layer.fuse_gate_up = False
        expected = layer(x, indices)
        layer.fuse_gate_up = True
        actual = layer(x, indices)
        mx.eval(expected, actual)

        assert_allclose(self, actual, expected)

    def test_quantized_gate_up_fusion_falls_back_without_building_cache(self):
        mx.random.seed(2)
        layer = SwitchGLU(64, 64, 4, bias=True)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=32, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=32, bits=4)
        layer.eval()
        x = mx.random.normal((8, 64))
        indices = mx.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]])

        layer.fuse_gate_up = False
        expected = layer(x, indices)
        layer.fuse_gate_up = True
        actual = layer(x, indices)
        mx.eval(expected, actual)

        assert_allclose(self, actual, expected, rtol=1e-4, atol=1e-4)
        self.assertIsNone(layer._fused_gate_up_cache)

    def test_training_mode_falls_back_without_building_fused_cache(self):
        layer = SwitchGLU(16, 32, 4, fuse_gate_up=True)
        layer.train()
        x = mx.random.normal((4, 16))
        indices = mx.array([[0, 1], [1, 2], [2, 3], [3, 0]])

        layer(x, indices)

        self.assertIsNone(layer._fused_gate_up_cache)

    def test_fused_cache_does_not_add_parameters(self):
        mx.random.seed(3)
        layer = SwitchGLU(16, 32, 4, bias=True, fuse_gate_up=True)
        layer.eval()
        before = [key for key, _ in tree_flatten(layer.parameters())]
        x = mx.random.normal((4, 16))
        indices = mx.array([[0, 1], [1, 2], [2, 3], [3, 0]])

        layer(x, indices)
        after = [key for key, _ in tree_flatten(layer.parameters())]

        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
