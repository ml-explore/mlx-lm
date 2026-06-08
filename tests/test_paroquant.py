# Copyright © 2025 Apple Inc.
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.paroquant import (
    RotateQuantizedLinear,
    RotateSwitchGLU,
    apply_rotation,
    pack_pairs,
)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
from mlx_lm.utils import (
    _patch_paro_layers,
    _stack_paro_moe_experts,
    _transform_paro_weights,
)


def _valid_pairs(krot, in_features, group_size):
    """Within-group perfect matching: thread t handles disjoint pair (2t, 2t+1)."""
    num_groups = in_features // group_size
    base = np.arange(group_size, dtype=np.int16)
    return mx.array(np.tile(base, (krot, num_groups)))


def _random_valid_pairs(rng, krot, in_features, group_size):
    """Random within-group perfect matching (a random permutation per group).

    Each group stores a permutation of [0, group_size); consecutive entries
    form the disjoint (i, j) pairs, so lanes never alias -> deterministic.
    """
    num_groups = in_features // group_size
    pairs = np.empty((krot, in_features), dtype=np.int16)
    for k in range(krot):
        for g in range(num_groups):
            pairs[k, g * group_size : (g + 1) * group_size] = rng.permutation(
                group_size
            )
    return mx.array(pairs)


def _numpy_rotation(x, theta, channel_scales, pairs, group_size):
    """Faithful reference that reads the (i, j) local indices out of ``pairs``."""
    krot, half_hidden = theta.shape
    in_f = half_hidden * 2
    num_groups = in_f // group_size
    half_gs = group_size // 2
    cos, sin = np.cos(theta), np.sin(theta)
    pairs = np.asarray(pairs)
    xr = x.astype(np.float64) * channel_scales.astype(np.float64)
    for k in range(krot):
        for g in range(num_groups):
            ci = g * group_size
            for t in range(half_gs):
                i_local = int(pairs[k, ci + 2 * t])
                j_local = int(pairs[k, ci + 2 * t + 1])
                col = g * half_gs + t
                c, s = cos[k, col], sin[k, col]
                a, b = xr[:, ci + i_local].copy(), xr[:, ci + j_local].copy()
                xr[:, ci + i_local] = a * c + b * s
                xr[:, ci + j_local] = b * c - a * s
    return xr


class TestParoQuant(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.gs, self.krot, self.bits = 128, 8, 4
        self.in_f, self.out_f = 512, 256

    def _check_rotation(self, pairs):
        theta = self.rng.standard_normal((self.krot, self.in_f // 2)).astype(np.float32)
        cscales = self.rng.standard_normal((self.in_f,)).astype(np.float32)
        x = self.rng.standard_normal((4, self.in_f)).astype(np.float32)
        out = apply_rotation(
            mx.array(x),
            pack_pairs(pairs, self.gs),
            mx.cos(mx.array(theta)),
            mx.sin(mx.array(theta)),
            mx.array(cscales),
            self.in_f,
            self.krot,
            self.gs,
        )
        ref = _numpy_rotation(x, theta, cscales, np.array(pairs), self.gs)
        self.assertTrue(np.allclose(np.array(out), ref, atol=1e-4))

    def test_rotation_matches_numpy(self):
        # Fixed (2t, 2t+1) matching.
        self._check_rotation(_valid_pairs(self.krot, self.in_f, self.gs))

    def test_rotation_random_pairs(self):
        # Random within-group matchings exercise arbitrary (i, j) index values.
        self._check_rotation(
            _random_valid_pairs(self.rng, self.krot, self.in_f, self.gs)
        )

    def test_rotate_quantized_linear_forward(self):
        layer = RotateQuantizedLinear(
            self.in_f,
            self.out_f,
            bias=True,
            group_size=self.gs,
            bits=self.bits,
            krot=self.krot,
        )
        layer.theta = mx.array(
            self.rng.standard_normal(layer.theta.shape).astype(np.float32)
        )
        layer.pairs = _valid_pairs(self.krot, self.in_f, self.gs)
        layer.channel_scales = mx.array(
            self.rng.standard_normal((1, self.in_f)).astype(np.float32)
        )
        layer.weight = mx.array(
            self.rng.integers(0, 2**32, size=layer.weight.shape, dtype=np.uint32)
        )
        layer.scales = mx.array(
            (self.rng.standard_normal(layer.scales.shape) * 0.05).astype(np.float32)
        )
        layer.biases = mx.array(
            (self.rng.standard_normal(layer.biases.shape) * 0.05).astype(np.float32)
        )
        layer.bias = mx.array(
            self.rng.standard_normal((self.out_f,)).astype(np.float32)
        )

        x = mx.array(self.rng.standard_normal((2, 7, self.in_f)).astype(np.float32))
        y = layer(x)
        mx.eval(y)
        self.assertEqual(y.shape, (2, 7, self.out_f))
        self.assertTrue(bool(mx.all(mx.isfinite(y)).item()))

    def test_transform_paro_weights_awq_packed(self):
        # Public ParoQuant checkpoints ship AWQ-packed weights + rotation params.
        prefix = "model.layers.0.self_attn.q_proj"
        pf = 32 // self.bits
        n_groups = self.in_f // self.gs
        weights = {
            f"{prefix}.qweight": mx.array(
                self.rng.integers(
                    0, 2**31, size=(self.in_f, self.out_f // pf), dtype=np.int32
                )
            ),
            f"{prefix}.qzeros": mx.array(
                self.rng.integers(
                    0, 2**31, size=(n_groups, self.out_f // pf), dtype=np.int32
                )
            ),
            f"{prefix}.scales": mx.array(
                (self.rng.standard_normal((n_groups, self.out_f)) * 0.02).astype(
                    np.float16
                )
            ),
            f"{prefix}.theta": mx.array(
                self.rng.standard_normal((self.krot, self.in_f // 2)).astype(np.float16)
            ),
            f"{prefix}.pairs": _valid_pairs(self.krot, self.in_f, self.gs),
            f"{prefix}.channel_scales": mx.array(
                self.rng.standard_normal((1, self.in_f)).astype(np.float16)
            ),
            "model.embed_tokens.weight": mx.zeros((100, self.in_f), dtype=mx.float16),
        }
        out = _transform_paro_weights(
            weights, {"bits": self.bits, "group_size": self.gs}
        )

        self.assertEqual(out[f"{prefix}.weight"].dtype, mx.uint32)
        self.assertEqual(
            out[f"{prefix}.weight"].shape, (self.out_f, self.in_f * self.bits // 32)
        )
        self.assertEqual(out[f"{prefix}.scales"].shape, (self.out_f, n_groups))
        self.assertEqual(out[f"{prefix}.biases"].shape, (self.out_f, n_groups))
        # rotation params pass through untouched
        for r in ("theta", "pairs", "channel_scales"):
            self.assertIn(f"{prefix}.{r}", out)
        self.assertNotIn(f"{prefix}.qweight", out)
        self.assertIn("model.embed_tokens.weight", out)

    def test_patch_dense_layer(self):
        class Tiny(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.proj = nn.Linear(in_f, out_f, bias=False)

        model = Tiny(self.in_f, self.out_f)
        weights = {
            "proj.theta": mx.zeros((self.krot, self.in_f // 2)),
            "proj.weight": mx.zeros(
                (self.out_f, self.in_f * self.bits // 32), dtype=mx.uint32
            ),
        }
        _patch_paro_layers(model, weights, self.bits, self.gs)
        self.assertIsInstance(model.proj, RotateQuantizedLinear)
        self.assertEqual(model.proj.bits, self.bits)
        self.assertEqual(model.proj.group_size, self.gs)

    def test_patch_moe_switchglu(self):
        n_experts, hidden, ffn = 4, 256, 512
        glu = SwitchGLU(hidden, ffn, n_experts)
        weights = {
            "switch_mlp.gate_up_rot_theta": mx.zeros((self.krot, hidden // 2)),
            "switch_mlp.gate_up_rot_pairs": _valid_pairs(self.krot, hidden, self.gs),
            "switch_mlp.gate_up_rot_channel_scales": mx.ones((1, hidden)),
            "switch_mlp.down_rot_theta": mx.zeros((self.krot, ffn // 2)),
            "switch_mlp.down_rot_pairs": _valid_pairs(self.krot, ffn, self.gs),
            "switch_mlp.down_rot_channel_scales": mx.ones((1, ffn)),
        }

        class Wrap(nn.Module):
            def __init__(self, glu):
                super().__init__()
                self.switch_mlp = glu

        model = Wrap(glu)
        _patch_paro_layers(model, weights, self.bits, self.gs)
        self.assertIsInstance(model.switch_mlp, RotateSwitchGLU)

    def test_moe_stack_and_forward(self):
        # Per-expert quantized weights -> _stack_paro_moe_experts -> patched
        # RotateSwitchGLU over QuantizedSwitchLinear experts -> forward smoke.
        n_experts, hidden, ffn = 4, 256, 512
        base = "mlp"
        weights = {}
        for proj, (out_d, in_d) in [
            ("gate_proj", (ffn, hidden)),
            ("up_proj", (ffn, hidden)),
            ("down_proj", (hidden, ffn)),
        ]:
            for e in range(n_experts):
                w = mx.random.normal((out_d, in_d)) * 0.05
                qw, sc, bi = mx.quantize(w, group_size=self.gs, bits=self.bits)
                weights[f"{base}.experts.{e}.{proj}.weight"] = qw
                weights[f"{base}.experts.{e}.{proj}.scales"] = sc
                weights[f"{base}.experts.{e}.{proj}.biases"] = bi
        for rot, dim in [("gate_up_rot", hidden), ("down_rot", ffn)]:
            weights[f"{base}.switch_mlp.{rot}_theta"] = mx.zeros((self.krot, dim // 2))
            weights[f"{base}.switch_mlp.{rot}_pairs"] = _valid_pairs(
                self.krot, dim, self.gs
            )
            weights[f"{base}.switch_mlp.{rot}_channel_scales"] = mx.ones((1, dim))

        weights = _stack_paro_moe_experts(weights)
        # 4 experts stacked into one (E, out, in/pack) tensor per projection
        gw = weights[f"{base}.switch_mlp.gate_proj.weight"]
        self.assertEqual(gw.shape[0], n_experts)
        self.assertNotIn(f"{base}.experts.0.gate_proj.weight", weights)

        class Wrap(nn.Module):
            def __init__(self, glu):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.switch_mlp = glu

        model = Wrap(SwitchGLU(hidden, ffn, n_experts))
        _patch_paro_layers(model, weights, self.bits, self.gs)
        self.assertIsInstance(model.mlp.switch_mlp, RotateSwitchGLU)
        self.assertIsInstance(model.mlp.switch_mlp.gate_proj, QuantizedSwitchLinear)
        model.load_weights(list(weights.items()), strict=False)

        n_tok, top_k = 6, 2
        x = mx.random.normal((n_tok, hidden))
        idx = mx.array(self.rng.integers(0, n_experts, size=(n_tok, top_k)))
        y = model.mlp.switch_mlp(x, idx)
        mx.eval(y)
        self.assertEqual(y.shape, (n_tok, top_k, hidden))
        self.assertTrue(bool(mx.all(mx.isfinite(y)).item()))


if __name__ == "__main__":
    unittest.main()
