# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

import mlx_lm.models.gated_delta as gated_delta
from mlx_lm.models.gated_delta import (
    gated_delta_chunked,
    gated_delta_kernel,
    gated_delta_kernel_unpacked,
    gated_delta_kernel_xtree,
    gated_delta_ops,
    gated_delta_update,
    normalize_qk,
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
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
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
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, mask)
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
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_small_head_dim_generic_matches_ops_reference(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(1, 65, 4, 8, 64, 64, mx.bfloat16)
        y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state, None)
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
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

    def test_chunked_matches_ops_forward(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        cases = [
            (1, 16, 32, 128, 128),  # Qwen3.5/3.6 shape
            (1, 4, 8, 128, 128),  # fewer heads
            (2, 16, 32, 128, 128),  # batched
            (1, 8, 16, 128, 64),  # Dv != Dk
        ]
        for B, Hk, Hv, Dk, Dv in cases:
            for T in (7, 64, 128, 130):
                for cs in (32, 64):
                    with self.subTest(B=B, Hv=Hv, T=T, cs=cs):
                        q, k, v, g, beta, state = self._inputs(
                            B, T, Hk, Hv, Dk, Dv, mx.float32
                        )
                        y_c, s_c = gated_delta_chunked(
                            q, k, v, g, beta, state, chunk_size=cs
                        )
                        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state)
                        mx.eval(y_c, s_c, y_r, s_r)
                        self.assertLess(_rel_l2(y_c, y_r), 1e-4)
                        self.assertLess(_rel_l2(s_c, s_r), 1e-4)

    def test_chunked_matches_ops_backward(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(1, 64, 16, 32, 128, 128, mx.float32)

        def loss_ops(q, k, v, g, beta):
            return gated_delta_ops(q, k, v, g, beta, state)[0].sum()

        def loss_chunked(q, k, v, g, beta):
            return gated_delta_chunked(q, k, v, g, beta, state, chunk_size=32)[0].sum()

        (_,), gr = mx.vjp(loss_ops, [q, k, v, g, beta], [mx.array(1.0)])
        (_,), gc = mx.vjp(loss_chunked, [q, k, v, g, beta], [mx.array(1.0)])
        mx.eval(*gr, *gc)
        for name, a, b in zip(("q", "k", "v", "g", "beta"), gc, gr):
            with self.subTest(grad=name):
                self.assertLess(_rel_l2(a, b), 1e-4)

    def test_chunked_allows_neg_eigval_beta(self):
        # allow_neg_eigval puts beta in [0, 2]; the UT transform must stay stable.
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(1, 128, 16, 32, 128, 128, mx.float32)
        beta = beta * 2.0
        y_c, s_c = gated_delta_chunked(q, k, v, g, beta, state, chunk_size=64)
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state)
        mx.eval(y_c, s_c, y_r, s_r)
        self.assertLess(_rel_l2(y_c, y_r), 1e-4)
        self.assertLess(_rel_l2(s_c, s_r), 1e-4)

    def test_update_training_routes_to_chunked_and_differentiable(self):
        # gated_delta_update(use_kernel=False) must match the kernel path and,
        # unlike the kernel, be differentiable (the point of the chunked path).
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        mx.random.seed(0)
        B, T, Hk, Hv, Dk, Dv = 1, 128, 16, 32, 128, 128
        q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
        k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
        q, k = normalize_qk(q, k, inv_scale=Dk**-0.5, eps=1e-6)
        v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
        a = mx.random.normal((B, T, Hv))
        b = mx.random.normal((B, T, Hv))
        A_log = mx.log(mx.random.uniform(low=0.01, high=16, shape=(Hv,)))
        dt_bias = mx.ones((Hv,))

        y_k, s_k = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=True)
        y_c, s_c = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False)
        mx.eval(y_k, s_k, y_c, s_c)
        self.assertLess(_rel_l2(y_c, y_k), 5e-3)
        self.assertLess(_rel_l2(s_c, s_k), 5e-3)

        def loss(q, k, v, a, b):
            return gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False)[
                0
            ].sum()

        (_,), grads = mx.vjp(loss, [q, k, v, a, b], [mx.array(1.0)])
        mx.eval(*grads)
        self.assertTrue(all((gr != 0).sum().item() > 0 for gr in grads))

    def test_chunked_numerical_extremes(self):
        # Chunked must track ops across decay/beta/value extremes and a large
        # single chunk -- the operating envelope a real training run can hit.
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        q, k, v, g, beta, state = self._inputs(1, 128, 16, 32, 128, 128, mx.float32)

        variants = {
            "strong_decay": (
                q,
                k,
                v,
                mx.exp(-mx.random.uniform(shape=g.shape) * 5),
                beta,
            ),
            "extreme_decay": (
                q,
                k,
                v,
                mx.exp(-mx.random.uniform(shape=g.shape) * 20),
                beta,
            ),
            "beta_zero": (q, k, v, g, mx.zeros_like(beta)),
            "beta_double": (q, k, v, g, beta * 2.0),  # allow_neg_eigval max
            "tiny_v": (q, k, v * 1e-4, g, beta),
            "huge_v": (q, k, v * 1e4, g, beta),
        }
        for name, (qq, kk, vv, gg, bb) in variants.items():
            with self.subTest(case=name):
                y_c, s_c = gated_delta_chunked(qq, kk, vv, gg, bb, state, chunk_size=64)
                y_r, s_r = gated_delta_ops(qq, kk, vv, gg, bb, state)
                mx.eval(y_c, s_c, y_r, s_r)
                self.assertLess(_rel_l2(y_c, y_r), 1e-3)
                self.assertLess(_rel_l2(s_c, s_r), 1e-3)

        # a single chunk spanning the whole sequence (Newton-Schulz at C=256)
        q2, k2, v2, g2, beta2, st2 = self._inputs(1, 256, 16, 32, 128, 128, mx.float32)
        y_c, s_c = gated_delta_chunked(q2, k2, v2, g2, beta2, st2, chunk_size=256)
        y_r, s_r = gated_delta_ops(q2, k2, v2, g2, beta2, st2)
        mx.eval(y_c, s_c, y_r, s_r)
        self.assertLess(_rel_l2(y_c, y_r), 1e-3)
        self.assertLess(_rel_l2(s_c, s_r), 1e-3)

    def test_update_decode_and_mask_fall_back_to_ops(self):
        # T==1 (decode) and masked inputs must NOT take the chunked path; they
        # fall back and stay consistent with the kernel path.
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")

        def make(T):
            mx.random.seed(2)
            q = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16)
            k = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16)
            q, k = normalize_qk(q, k, inv_scale=128**-0.5, eps=1e-6)
            v = mx.random.normal((1, T, 32, 128)).astype(mx.bfloat16)
            a = mx.random.normal((1, T, 32))
            b = mx.random.normal((1, T, 32))
            A_log = mx.log(mx.random.uniform(low=0.01, high=16, shape=(32,)))
            dt_bias = mx.ones((32,))
            return q, k, v, a, b, A_log, dt_bias

        # decode: T == 1 falls back (guard q.shape[1] > 1)
        args = make(1)
        yf, _ = gated_delta_update(*args, use_kernel=False)
        yt, _ = gated_delta_update(*args, use_kernel=True)
        mx.eval(yf, yt)
        self.assertLess(_rel_l2(yf, yt), 5e-3)

        # mask present falls back (guard mask is None)
        q, k, v, a, b, A_log, dt_bias = make(64)
        mask = mx.arange(64)[None] < mx.array([[50]])
        yf, _ = gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, mask=mask, use_kernel=False
        )
        yt, _ = gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, mask=mask, use_kernel=True
        )
        valid = mask[..., None, None]
        yf = mx.where(valid, yf, 0)
        yt = mx.where(valid, yt, 0)
        mx.eval(yf, yt)
        self.assertLess(_rel_l2(yf, yt), 5e-3)


if __name__ == "__main__":
    unittest.main()
