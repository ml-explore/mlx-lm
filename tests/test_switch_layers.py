# Copyright (c) 2025, the contributors. All rights reserved.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)


def _reference_unfused(layer, x, indices):
    """Compute SwitchGLU output using separate gate and up projections."""
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)

    x_up = layer.up_proj(x, idx, sorted_indices=do_sort)
    x_gate = layer.gate_proj(x, idx, sorted_indices=do_sort)
    x = layer.down_proj(
        layer.activation(x_up, x_gate),
        idx,
        sorted_indices=do_sort,
    )

    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


def _param_keys(module):
    return sorted(k for k, _ in tree_flatten(module.parameters()))


def _make_indices_8():
    return mx.array([[0, 1], [1, 2], [0, 3], [2, 1], [3, 0], [1, 2], [0, 1], [2, 3]])


class TestSwitchGLUFusion(unittest.TestCase):
    def test_fused_quantized_matches_unfused(self):
        """Fused quantized gather_qmm matches the unfused two-call path."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=64, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=64, bits=4)
        layer.down_proj = layer.down_proj.to_quantized(group_size=64, bits=4)
        layer.eval()

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        expected = _reference_unfused(layer, x, indices)
        actual = layer(x, indices)
        mx.eval(expected, actual)

        self.assertTrue(mx.allclose(expected, actual, atol=1e-5))

    def test_fused_nonquantized_matches_unfused(self):
        """Fused non-quantized gather_mm matches the unfused two-call path."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4)
        layer.eval()

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        expected = _reference_unfused(layer, x, indices)
        actual = layer(x, indices)
        mx.eval(expected, actual)

        self.assertTrue(mx.allclose(expected, actual, atol=1e-5))

    def test_fused_quantized_with_sorting(self):
        """Fused path works when index sorting is triggered."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 8)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=64, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=64, bits=4)
        layer.down_proj = layer.down_proj.to_quantized(group_size=64, bits=4)
        layer.eval()

        # 32 tokens x top-2 = 64 indices, triggers sorting path
        x = mx.random.normal((32, 64))
        indices = mx.random.randint(0, 8, (32, 2))

        expected = _reference_unfused(layer, x, indices)
        actual = layer(x, indices)
        mx.eval(expected, actual)

        self.assertTrue(mx.allclose(expected, actual, atol=1e-5))

    def test_fused_repeated_calls(self):
        """Repeated forward passes produce consistent results after fusion."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=64, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=64, bits=4)
        layer.down_proj = layer.down_proj.to_quantized(group_size=64, bits=4)
        layer.eval()

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        expected = _reference_unfused(layer, x, indices)
        result1 = layer(x, indices)
        result2 = layer(x, indices)
        mx.eval(expected, result1, result2)

        self.assertTrue(mx.allclose(expected, result1, atol=1e-5))
        self.assertTrue(mx.allclose(result1, result2))

    def test_fused_nonquantized_with_bias(self):
        """Fused non-quantized path handles linear bias correctly."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4, bias=True)
        layer.eval()

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        expected = _reference_unfused(layer, x, indices)
        actual = layer(x, indices)
        mx.eval(expected, actual)

        self.assertTrue(mx.allclose(expected, actual, atol=1e-5))

    def test_no_fusion_during_training(self):
        """Fusion is skipped when self.training is True."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=64, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=64, bits=4)
        layer.down_proj = layer.down_proj.to_quantized(group_size=64, bits=4)
        layer.train()

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        result = layer(x, indices)
        mx.eval(result)

        # Fused cache should not have been built
        self.assertFalse(hasattr(layer, "_fused"))
        # Original projections accessible
        self.assertIsInstance(layer.gate_proj, QuantizedSwitchLinear)
        self.assertIsInstance(layer.up_proj, QuantizedSwitchLinear)

    def test_quant_config_mismatch_fallback(self):
        """Mismatched quantization config falls back to unfused path."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=64, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=32, bits=4)
        layer.down_proj = layer.down_proj.to_quantized(group_size=64, bits=4)
        layer.eval()

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        expected = _reference_unfused(layer, x, indices)
        actual = layer(x, indices)
        mx.eval(expected, actual)

        self.assertFalse(layer._fused)
        self.assertTrue(mx.allclose(expected, actual, atol=1e-5))

    def test_asymmetric_bias_fallback(self):
        """Asymmetric bias configuration falls back to unfused path."""
        mx.random.seed(42)
        # Create with bias, then remove bias from up_proj only
        layer = SwitchGLU(64, 128, 4, bias=True)
        layer.eval()
        if "bias" in layer.up_proj:
            del layer.up_proj.bias

        x = mx.random.normal((8, 64))
        indices = _make_indices_8()

        result = layer(x, indices)
        mx.eval(result)

        self.assertFalse(layer._fused)
        self.assertFalse(mx.any(mx.isnan(result)).item())

    def test_param_keys_preserved_after_fusion(self):
        """Parameter tree keys are unchanged after fusion."""
        mx.random.seed(42)
        layer = SwitchGLU(64, 128, 4)
        layer.gate_proj = layer.gate_proj.to_quantized(group_size=64, bits=4)
        layer.up_proj = layer.up_proj.to_quantized(group_size=64, bits=4)
        layer.down_proj = layer.down_proj.to_quantized(group_size=64, bits=4)
        layer.eval()

        keys_before = _param_keys(layer)

        # Trigger fusion
        x = mx.random.normal((8, 64))
        indices = _make_indices_8()
        _ = layer(x, indices)
        mx.eval(_)

        keys_after = _param_keys(layer)
        self.assertEqual(keys_before, keys_after)


if __name__ == "__main__":
    unittest.main()
