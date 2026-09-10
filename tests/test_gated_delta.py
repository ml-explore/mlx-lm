# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

import mlx_lm.models.gated_delta as gated_delta
from mlx_lm.models.gated_delta import (
    gated_delta_chunkwise,
    gated_delta_kernel,
    gated_delta_kernel_unpacked,
    gated_delta_kernel_xtree,
    gated_delta_sequential,
)


def _normed(shape, D, dtype):
    x = mx.random.normal(shape)
    return (mx.fast.rms_norm(x, None, 1e-6) * D**-0.5).astype(dtype)


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-9)).item()


class TestGatedDelta(unittest.TestCase):
    def setUp(self):
        self._packed = gated_delta._ENABLE_GDN_PACKED
        gated_delta._ENABLE_GDN_PACKED = True

    def tearDown(self):
        gated_delta._ENABLE_GDN_PACKED = self._packed

    def test_kill_switch_restores_original_kernel(self):
        # MLX_GDN_PACKED=0 routes back to the pre-existing simd_sum kernel.
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        gated_delta._ENABLE_GDN_PACKED = False
        args = self._inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
        y1, s1 = gated_delta_kernel(*args, None)
        y2, s2 = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y1, s1, y2, s2)
        self.assertTrue(mx.array_equal(y1, y2))
        self.assertTrue(mx.array_equal(s1, s2))

    def _inputs(self, B, T, Hk, Hv, Dk, Dv, dtype):
        mx.random.seed(3)
        q = _normed((B, T, Hk, Dk), Dk, dtype)
        k = _normed((B, T, Hk, Dk), Dk, dtype)
        v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
        # decays in (0, 1], as produced by compute_g
        g = mx.exp(-mx.random.uniform(shape=(B, T, Hv)) * 0.2).astype(mx.float32)
        beta = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
        state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
        mx.eval(q, k, v, g, beta, state)
        return q, k, v, g, beta, state

    def test_packed_matches_unpacked(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        cases = [
            # B, Hk, Hv, Dk, Dv, dtype
            (1, 16, 32, 128, 128, mx.bfloat16),  # Qwen3.5/3.6 shape
            (2, 16, 32, 128, 128, mx.bfloat16),  # batched
            (1, 4, 8, 128, 128, mx.bfloat16),  # fewer heads
            (1, 8, 8, 128, 128, mx.bfloat16),  # Hv == Hk
            (1, 16, 32, 128, 128, mx.float16),
            (1, 16, 32, 128, 128, mx.float32),
            (1, 8, 16, 128, 64, mx.bfloat16),  # Dv != Dk
            (3, 2, 8, 128, 256, mx.bfloat16),  # larger Dv
        ]
        for B, Hk, Hv, Dk, Dv, dtype in cases:
            for T in (1, 7, 64, 257):  # decode, ragged, aligned, spilling
                with self.subTest(B=B, Hk=Hk, Hv=Hv, Dv=Dv, dtype=dtype, T=T):
                    args = self._inputs(B, T, Hk, Hv, Dk, Dv, dtype)
                    y_p, s_p = gated_delta_kernel(*args, None)
                    y_x, s_x = gated_delta_kernel_xtree(*args, None)
                    mx.eval(y_p, s_p, y_x, s_x)
                    # The packed kernel reproduces the comparator's explicit
                    # reduction tree, so this holds on any device
                    # independently of how simd_sum lowers.
                    self.assertTrue(mx.array_equal(y_p, y_x))
                    self.assertTrue(mx.array_equal(s_p, s_x))

    def test_packed_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        q, k, v, g, beta, state = self._inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
        y_p, s_p = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_sequential(q, k, v, g, beta, state, None)
        mx.eval(y_p, s_p, y_r, s_r)
        self.assertLess(_rel_l2(y_p, y_r), 2e-3)
        self.assertLess(_rel_l2(s_p, s_r), 2e-3)

    def test_explicit_tree_matches_simd_sum_kernel(self):
        # On all current Apple GPUs simd_sum lowers to the same ascending
        # butterfly the comparator writes out, so the packed kernel is also
        # bit-identical to the pre-existing kernel. If this ever fails on a
        # new device or toolchain, the packed default should be revisited
        # (its contract vs the comparator still holds).
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        args = self._inputs(1, 257, 16, 32, 128, 128, mx.bfloat16)
        y_x, s_x = gated_delta_kernel_xtree(*args, None)
        y_u, s_u = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y_x, s_x, y_u, s_u)
        self.assertTrue(mx.array_equal(y_x, y_u))
        self.assertTrue(mx.array_equal(s_x, s_u))

    def test_masked_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(2, 33, 8, 16, 128, 128, mx.bfloat16)
        mask = mx.arange(33)[None] < mx.array([[29], [17]])
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, mask)
        y_r, s_r = gated_delta_sequential(q, k, v, g, beta, state, mask)
        mx.eval(y_k, s_k, y_r, s_r)
        # Outputs at padded positions are unspecified (the kernel zeros
        # them, the ops reference does not); compare valid positions only.
        valid = mask[..., None, None]
        y_k = mx.where(valid, y_k, 0)
        y_r = mx.where(valid, y_r, 0)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_vector_gate_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, _, beta, state = self._inputs(1, 65, 4, 8, 128, 128, mx.bfloat16)
        g = mx.exp(-mx.random.uniform(shape=(1, 65, 8, 128)) * 0.2).astype(mx.float32)
        mx.eval(g)
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_sequential(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_small_head_dim_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(1, 65, 4, 8, 64, 64, mx.bfloat16)
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_sequential(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_unsupported_shapes_fall_back(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        # Dk != 128 must take the general kernel and stay exact against it.
        q, k, v, g, beta, state = self._inputs(1, 32, 4, 8, 64, 64, mx.bfloat16)
        y_p, s_p = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_u, s_u = gated_delta_kernel_unpacked(q, k, v, g, beta, state, None)
        mx.eval(y_p, s_p, y_u, s_u)
        self.assertTrue(mx.array_equal(y_p, y_u))
        self.assertTrue(mx.array_equal(s_p, s_u))

        # A padding mask also forces the general kernel.
        q, k, v, g, beta, state = self._inputs(1, 16, 16, 32, 128, 128, mx.bfloat16)
        mask = mx.ones((1, 16), dtype=mx.bool_)
        y_p, s_p = gated_delta_kernel(q, k, v, g, beta, state, mask)
        y_u, s_u = gated_delta_kernel_unpacked(q, k, v, g, beta, state, mask)
        mx.eval(y_p, s_p, y_u, s_u)
        self.assertTrue(mx.array_equal(y_p, y_u))
        self.assertTrue(mx.array_equal(s_p, s_u))


class TestGatedDeltaChunkwise(unittest.TestCase):
    """The chunkwise training path must agree with the sequential reference."""

    def _inputs(self, B, T, Hk, Hv, Dk, Dv, decay, dtype=mx.float32):
        mx.random.seed(7)
        q = _normed((B, T, Hk, Dk), Dk, dtype)
        k = _normed((B, T, Hk, Dk), Dk, dtype)
        v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
        g = mx.exp(-mx.random.uniform(shape=(B, T, Hv)) * decay).astype(mx.float32)
        beta = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
        state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
        mx.eval(q, k, v, g, beta, state)
        return q, k, v, g, beta, state

    def test_matches_ops(self):
        # decay 0.2 is the mild regime the kernels are tested at; 4.0 drives the
        # cumulative products small, where dividing by them would overflow.
        for decay in (0.2, 4.0):
            for T, chunk in ((128, 64), (96, 64), (64, 16)):  # 96 exercises padding
                args = self._inputs(1, T, 8, 16, 64, 64, decay)
                y_ref, s_ref = gated_delta_sequential(*args, None)
                y, s = gated_delta_chunkwise(*args, None, chunk_size=chunk)
                mx.eval(y_ref, s_ref, y, s)
                self.assertLess(_rel_l2(y, y_ref), 1e-5, f"decay={decay} T={T}")
                self.assertLess(_rel_l2(s, s_ref), 1e-5, f"decay={decay} T={T}")

    def test_grouped_query_heads(self):
        args = self._inputs(2, 64, 4, 16, 64, 64, 0.5)  # Hv // Hk == 4
        y_ref, s_ref = gated_delta_sequential(*args, None)
        y, s = gated_delta_chunkwise(*args, None, chunk_size=32)
        mx.eval(y_ref, s_ref, y, s)
        self.assertLess(_rel_l2(y, y_ref), 1e-5)
        self.assertLess(_rel_l2(s, s_ref), 1e-5)

    def test_gradients_match_ops(self):
        q, k, v, g, beta, state = self._inputs(1, 64, 2, 4, 32, 32, 0.5)
        w = mx.random.normal((1, 64, 4, 32))

        def make(fn, **kwargs):
            def loss(q, k, v, g, beta):
                y, s = fn(q, k, v, g, beta, state, None, **kwargs)
                return (y * w).sum() + (s * s).sum()

            return loss

        argnums = (0, 1, 2, 3, 4)
        ref = mx.grad(make(gated_delta_sequential), argnums=argnums)(q, k, v, g, beta)
        got = mx.grad(make(gated_delta_chunkwise, chunk_size=16), argnums=argnums)(
            q, k, v, g, beta
        )
        mx.eval(ref, got)
        for a, b in zip(ref, got):
            self.assertLess(_rel_l2(b, a), 1e-4)


if __name__ == "__main__":
    unittest.main()
