"""Tests for GPT-OSS sanitize() supporting both openai/MXFP4 and unsloth/HF BF16 layouts."""
import unittest

import mlx.core as mx

from mlx_lm.models.gpt_oss import Model, ModelArgs


def _make_args(hidden_size=2880, intermediate_size=2880, num_local_experts=32):
    return ModelArgs(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
    )


class TestGptOssSanitize(unittest.TestCase):
    def test_unsloth_bf16_shapes_and_weight_suffix(self):
        """Layout (b): (E, D, 2H) experts.gate_up_proj + missing .weight suffix."""
        E, D, H = 32, 2880, 2880
        weights = {
            "model.layers.0.mlp.experts.gate_up_proj":      mx.zeros((E, D, 2 * H), dtype=mx.bfloat16),
            "model.layers.0.mlp.experts.down_proj":         mx.zeros((E, D, D),     dtype=mx.bfloat16),
            "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((E, 2 * H),    dtype=mx.bfloat16),
            "model.layers.0.mlp.experts.down_proj_bias":    mx.zeros((E, D),        dtype=mx.bfloat16),
        }
        m = Model(_make_args())
        s = m.sanitize(weights)
        self.assertEqual(s["model.layers.0.mlp.experts.gate_proj.weight"].shape, (E, H, D))
        self.assertEqual(s["model.layers.0.mlp.experts.up_proj.weight"].shape,   (E, H, D))
        self.assertEqual(s["model.layers.0.mlp.experts.down_proj.weight"].shape, (E, D, H))
        self.assertEqual(s["model.layers.0.mlp.experts.gate_proj.bias"].shape, (E, H))
        self.assertEqual(s["model.layers.0.mlp.experts.up_proj.bias"].shape,   (E, H))

    def test_unsloth_bf16_interleaved_split_values(self):
        """Both layouts use interleaved gate/up split on the 2*H axis."""
        E, D, H = 2, 4, 3
        # Storage (E, D, 2H): along axis -1, mark even slots with 1.0 (gate)
        # and odd slots with 2.0 (up).
        v = mx.zeros((E, D, 2 * H), dtype=mx.float32)
        idx_even = mx.arange(0, 2 * H, 2)
        idx_odd = mx.arange(1, 2 * H, 2)
        for i in idx_even.tolist():
            v[:, :, i] = 1.0
        for i in idx_odd.tolist():
            v[:, :, i] = 2.0
        weights = {
            "model.layers.0.mlp.experts.gate_up_proj":      v,
            "model.layers.0.mlp.experts.down_proj":         mx.zeros((E, D, D), dtype=mx.float32),
            "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((E, 2 * H), dtype=mx.float32),
            "model.layers.0.mlp.experts.down_proj_bias":    mx.zeros((E, D), dtype=mx.float32),
        }
        args = _make_args(hidden_size=D, intermediate_size=H, num_local_experts=E)
        m = Model(args)
        s = m.sanitize(weights)
        g = s["model.layers.0.mlp.experts.gate_proj.weight"]
        u = s["model.layers.0.mlp.experts.up_proj.weight"]
        self.assertTrue(mx.all(g == 1.0).item())
        self.assertTrue(mx.all(u == 2.0).item())

    def test_already_sanitized_passthrough(self):
        E, D, H = 32, 2880, 2880
        weights = {
            "model.layers.0.mlp.experts.gate_proj.weight": mx.zeros((E, H, D), dtype=mx.bfloat16),
        }
        m = Model(_make_args())
        s = m.sanitize(weights)
        self.assertIs(s, weights)

    def test_unsloth_bf16_down_proj_values_unequal_dims(self):
        """down_proj values are transposed correctly when hidden_size != intermediate_size."""
        E, D, H = 2, 5, 3
        # unsloth down_proj layout: (E, intermediate=H, hidden=D).
        # Mark each (intermediate, hidden) cell with a unique value so a wrong
        # transpose is detectable.
        dp = mx.zeros((E, H, D), dtype=mx.float32)
        for i in range(H):
            for j in range(D):
                dp[:, i, j] = float(i * 10 + j)
        weights = {
            "model.layers.0.mlp.experts.gate_up_proj":      mx.zeros((E, D, 2 * H), dtype=mx.float32),
            "model.layers.0.mlp.experts.down_proj":         dp,
            "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((E, 2 * H),    dtype=mx.float32),
            "model.layers.0.mlp.experts.down_proj_bias":    mx.zeros((E, D),        dtype=mx.float32),
        }
        args = _make_args(hidden_size=D, intermediate_size=H, num_local_experts=E)
        m = Model(args)
        s = m.sanitize(weights)
        out = s["model.layers.0.mlp.experts.down_proj.weight"]
        # SwitchLinear convention: (E, out=hidden=D, in=intermediate=H).
        self.assertEqual(out.shape, (E, D, H))
        # Verify the swap was applied: out[:, j, i] == i*10 + j.
        for i in range(H):
            for j in range(D):
                self.assertEqual(out[0, j, i].item(), float(i * 10 + j))

    def test_unsloth_bf16_down_proj_values_equal_dims(self):
        """Regression guard: with hidden_size == intermediate_size the unsloth
        down_proj must still be transposed (the previous shape-only guard
        skipped it and emitted noise at inference time)."""
        E, D = 2, 4
        H = D  # equal dims — exactly the default GPT-OSS config.
        # Asymmetric values so a missed transpose is detectable.
        dp = mx.zeros((E, H, D), dtype=mx.float32)
        for i in range(H):
            for j in range(D):
                dp[:, i, j] = float(i * 10 + j)
        weights = {
            "model.layers.0.mlp.experts.gate_up_proj":      mx.zeros((E, D, 2 * H), dtype=mx.float32),
            "model.layers.0.mlp.experts.down_proj":         dp,
            "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((E, 2 * H),    dtype=mx.float32),
            "model.layers.0.mlp.experts.down_proj_bias":    mx.zeros((E, D),        dtype=mx.float32),
        }
        args = _make_args(hidden_size=D, intermediate_size=H, num_local_experts=E)
        m = Model(args)
        s = m.sanitize(weights)
        out = s["model.layers.0.mlp.experts.down_proj.weight"]
        self.assertEqual(out.shape, (E, D, H))
        for i in range(H):
            for j in range(D):
                self.assertEqual(out[0, j, i].item(), float(i * 10 + j))

    def test_openai_interleaved_layout_preserved(self):
        """Layout (a): (E, 2H, D) — split on axis -2 as before."""
        E, D, H = 32, 2880, 2880
        weights = {
            "model.layers.0.mlp.experts.gate_up_proj":      mx.zeros((E, 2 * H, D), dtype=mx.bfloat16),
            "model.layers.0.mlp.experts.down_proj":         mx.zeros((E, H, D),     dtype=mx.bfloat16),
            "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((E, 2 * H),    dtype=mx.bfloat16),
            "model.layers.0.mlp.experts.down_proj_bias":    mx.zeros((E, D),        dtype=mx.bfloat16),
        }
        m = Model(_make_args())
        s = m.sanitize(weights)
        self.assertEqual(s["model.layers.0.mlp.experts.gate_proj.weight"].shape, (E, H, D))
        self.assertEqual(s["model.layers.0.mlp.experts.up_proj.weight"].shape,   (E, H, D))
        self.assertEqual(s["model.layers.0.mlp.experts.down_proj.weight"].shape, (E, H, D))


if __name__ == "__main__":
    unittest.main()
