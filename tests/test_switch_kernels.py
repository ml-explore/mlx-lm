# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import switch_kernels
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU


class GeGLU(nn.Module):
    metal_act = 1

    def __call__(self, x, gate):
        return nn.gelu_approx(gate) * x


def quantized_glu(D, F, E, activation=None, bits=4):
    glu = SwitchGLU(D, F, E, activation=activation or SwiGLU())
    nn.quantize(glu, group_size=64, bits=bits)
    mx.eval(glu.parameters())
    return glu


def reference(glu, x, indices):
    """fp32 per-slot outputs on the dequantized weights."""
    w = [
        mx.dequantize(p.weight, p.scales, p.biases, group_size=64, bits=4)
        for p in (glu.gate_proj, glu.up_proj, glu.down_proj)
    ]
    wg, wu, wd = (a.astype(mx.float32)[indices] for a in w)
    x = x.astype(mx.float32)[..., None, :, None]
    g, u = (wg @ x).squeeze(-1), (wu @ x).squeeze(-1)
    a = nn.silu(g) if glu.activation.metal_act == 0 else nn.gelu_approx(g)
    return (wd @ (a * u)[..., None]).squeeze(-1)


def rel_err(a, b):
    return (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(b.flatten())).item()


@unittest.skipIf(not mx.metal.is_available(), "Metal only")
class TestSwitchKernels(unittest.TestCase):
    def test_matches_reference(self):
        mx.random.seed(0)
        shapes = [(256, 384, 16, 4), (2816, 704, 128, 8), (512, 192, 8, 2)]
        for D, F, E, k in shapes:
            for act in (SwiGLU(), GeGLU()):
                glu = quantized_glu(D, F, E, act)
                for dtype in (mx.bfloat16, mx.float16):
                    glu.set_dtype(dtype)
                    for T in (1, 2, 3, 8, 17):
                        x = mx.random.normal((1, T, D)).astype(dtype)
                        idx = mx.argsort(mx.random.uniform(shape=(1, T, E)), axis=-1)[
                            ..., :k
                        ]
                        if not switch_kernels.supported_load(T * k / E, D, F):
                            continue
                        y = switch_kernels.switch_glu(glu, x, idx)
                        self.assertIsNotNone(y)
                        self.assertEqual(y.shape, (1, T, k, D))
                        ref = reference(glu, x, idx)
                        err = rel_err(y.astype(mx.float32), ref)
                        self.assertLess(err, 1e-2, (D, F, E, k, T, dtype, err))

    def test_gradient_matches_generic_path(self):
        mx.random.seed(1)
        glu = quantized_glu(256, 384, 16)
        glu.set_dtype(mx.bfloat16)
        x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
        idx = mx.argsort(mx.random.uniform(shape=(1, 3, 16)), axis=-1)[..., :4]
        loss = lambda x: glu(x, idx).astype(mx.float32).sum()
        self.assertIsNotNone(switch_kernels.switch_glu(glu, x, idx))
        g_fast = mx.grad(loss)(x)
        kernels, switch_kernels._kernels = switch_kernels._kernels, None
        try:
            g_ref = mx.grad(loss)(x)
        finally:
            switch_kernels._kernels = kernels
        self.assertLess(
            rel_err(g_fast.astype(mx.float32), g_ref.astype(mx.float32)), 1e-2
        )

    def test_unsupported_falls_back(self):
        glu = quantized_glu(256, 384, 16, bits=8)
        x = mx.random.normal((1, 2, 256)).astype(mx.bfloat16)
        idx = mx.zeros((1, 2, 4), mx.uint32)
        self.assertIsNone(switch_kernels.switch_glu(glu, x, idx))
        glu = quantized_glu(256, 384, 16)
        # scales are fp32 here: the generic path would return fp32
        self.assertIsNone(switch_kernels.switch_glu(glu, x, idx))
        glu.set_dtype(mx.bfloat16)
        self.assertIsNotNone(switch_kernels.switch_glu(glu, x, idx))
        self.assertIsNone(switch_kernels.switch_glu(glu, x.astype(mx.float32), idx))
        many = mx.zeros((1, 512, 4), mx.uint32)
        self.assertIsNone(
            switch_kernels.switch_glu(glu, mx.zeros((1, 512, 256), mx.bfloat16), many)
        )
        # more experts per token than the T = 1 threadgroup allows
        self.assertIsNone(
            switch_kernels.switch_glu(glu, x, mx.zeros((1, 2, 9), mx.uint32))
        )

    def test_no_tokens(self):
        glu = quantized_glu(256, 384, 16)
        glu.set_dtype(mx.bfloat16)
        x, idx = mx.zeros((1, 0, 256), mx.bfloat16), mx.zeros((1, 0, 4), mx.uint32)
        self.assertEqual(glu(x, idx).shape, (1, 0, 4, 256))

    def test_nax_gpu_keeps_gather_qmm_rhs(self):
        glu = quantized_glu(256, 384, 16)
        glu.set_dtype(mx.bfloat16)
        x = lambda T: mx.zeros((1, T, 256), mx.bfloat16)
        idx = lambda T: mx.zeros((1, T, 4), mx.uint32)
        nax, switch_kernels._nax = switch_kernels._nax, True
        try:
            self.assertIsNotNone(switch_kernels.switch_glu(glu, x(15), idx(15)))
            self.assertIsNone(switch_kernels.switch_glu(glu, x(16), idx(16)))
        finally:
            switch_kernels._nax = nax


if __name__ == "__main__":
    unittest.main()
