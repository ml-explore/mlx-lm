# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.kernels import (
    expert_routing,
    norm_add_norm,
    post_expert_norm_combine,
    pre_norm,
    qk_norm,
    qk_norm_rope,
    rms_norm_sigmoid_gate,
)

TOL = {mx.float32: 1e-5, mx.bfloat16: 1e-2}

QK_ARGS = {
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 80,
    "eps": 1e-5,
    "query_scale": 4.9,
}


def _norm_inputs(dtype):
    # Not a multiple of the threadgroup size.
    D = 1000
    mx.random.seed(0)
    branch = (3 * mx.random.normal((2, 3, D))).astype(dtype)
    stream = 2 * mx.random.normal((2, 3, D))
    alpha = mx.random.uniform(0.5, 1.5, (D,))
    beta = mx.random.uniform(0.0, 1.0, (D,))
    post_weight = mx.random.uniform(0.5, 1.5, (D,)).astype(dtype)
    norm = nn.RMSNorm(D, eps=1e-5)
    norm.weight = mx.random.uniform(0.5, 1.5, (D,)).astype(dtype)
    return branch, stream, alpha, beta, post_weight, norm


def _norm_add_norm_ref(
    branch, stream, *, alpha, beta, post_weight, post_eps, next_norm
):
    x = branch if post_weight is None else post_weight * branch
    h = alpha * stream + beta * mx.fast.rms_norm(x, None, post_eps)
    if next_norm is None:
        return h, None
    return h, next_norm(h).astype(branch.dtype)


def _qkvg(dtype):
    mx.random.seed(1)
    H = QK_ARGS["n_heads"] + QK_ARGS["n_kv_heads"]
    x = 2 * mx.random.normal((2, 5, 2 * H * QK_ARGS["head_dim"]))
    return x.astype(dtype)


def _qk_norm_ref(qkvg, *, n_heads, n_kv_heads, head_dim, eps, query_scale):
    tokens = qkvg.shape[:-1]
    q_dim = n_heads * head_dim
    q = qkvg[..., :q_dim].reshape(*tokens, n_heads, head_dim)
    k = qkvg[..., q_dim : q_dim + n_kv_heads * head_dim]
    k = k.reshape(*tokens, n_kv_heads, head_dim)
    # q is scaled in fp32.
    q = mx.fast.rms_norm(q, None, eps).astype(mx.float32) * query_scale
    return q.astype(qkvg.dtype), mx.fast.rms_norm(k, None, eps)


def _rope(x, offset):
    x = x.swapaxes(1, 2)
    x = mx.fast.rope(
        x, x.shape[-1], traditional=True, base=10000.0, scale=1.0, offset=offset
    )
    return x.swapaxes(1, 2)


@unittest.skipIf(not mx.metal.is_available(), "The kernels need the Metal backend")
class TestKernels(unittest.TestCase):
    def _assert_close(self, out, ref, tol):
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, ref.dtype)
        out, ref = out.astype(mx.float32), ref.astype(mx.float32)
        self.assertTrue(mx.allclose(out, ref, atol=tol, rtol=tol))

    def test_norm_add_norm(self):
        for dtype in (mx.float32, mx.bfloat16):
            branch, stream, alpha, beta, post_weight, norm = _norm_inputs(dtype)
            for weight, next_norm in ((None, None), (post_weight, norm)):
                kwargs = {
                    "alpha": alpha,
                    "beta": beta,
                    "post_weight": weight,
                    "post_eps": 1e-6,
                    "next_norm": next_norm,
                }
                h, normed = norm_add_norm(branch, stream, **kwargs)
                ref_h, ref_normed = _norm_add_norm_ref(branch, stream, **kwargs)
                self._assert_close(h, ref_h, TOL[dtype])
                if next_norm is None:
                    self.assertIsNone(normed)
                else:
                    self._assert_close(normed, ref_normed, TOL[dtype])

    def test_pre_norm(self):
        for dtype in (mx.float32, mx.bfloat16):
            branch, stream, alpha, beta, post_weight, norm = _norm_inputs(dtype)
            h, normed = norm_add_norm(
                branch,
                stream,
                alpha=alpha,
                beta=beta,
                post_weight=post_weight,
                post_eps=1e-6,
                next_norm=norm,
            )
            out = pre_norm(h, norm, dtype=dtype)
            self.assertTrue(mx.array_equal(out, normed))

    def test_qk_norm(self):
        for dtype in (mx.float32, mx.bfloat16):
            qkvg = _qkvg(dtype)
            q, k = qk_norm(qkvg, **QK_ARGS)
            ref_q, ref_k = _qk_norm_ref(qkvg, **QK_ARGS)
            self._assert_close(q, ref_q, TOL[dtype])
            self._assert_close(k, ref_k, TOL[dtype])
            for offset in (7, mx.array([3, 11])):
                out_q, out_k = qk_norm_rope(
                    qkvg, **QK_ARGS, rope_base=10000.0, offset=offset
                )
                self._assert_close(out_q, _rope(q, offset), TOL[dtype])
                self._assert_close(out_k, _rope(k, offset), TOL[dtype])

    def test_expert_routing(self):
        mx.random.seed(2)
        # Some SIMD lanes have no expert.
        logits = 2 * mx.random.normal((2, 3, 40))
        bias = 0.1 * mx.random.normal((40,))
        inds, weights = expert_routing(
            logits, bias, top_k=4, multiplier=0.7, score_eps=1e-2, route_scale=2.5
        )

        # No ties for this seed.
        scores = mx.sigmoid(logits * 0.7)
        ref_inds = mx.argsort(-(scores + bias), axis=-1)[..., :4]
        ref_weights = mx.take_along_axis(scores, ref_inds, axis=-1)
        ref_weights = ref_weights / (ref_weights.sum(-1, keepdims=True) + 1e-2) * 2.5
        self.assertEqual(inds.dtype, mx.uint32)
        self.assertTrue(mx.array_equal(inds, ref_inds))
        self._assert_close(weights, ref_weights, TOL[mx.float32])

    def test_post_expert_norm_combine(self):
        # fp32 only: in bf16 the kernel rounds after each expert.
        mx.random.seed(3)
        experts = 3 * mx.random.normal((2, 3, 4, 1000))
        weights = mx.random.uniform(0.1, 1.0, (2, 3, 4))
        norm_weight = mx.random.uniform(0.5, 1.5, (1000,))
        out = post_expert_norm_combine(
            experts, weights, norm_weight=norm_weight, eps=1e-6
        )

        ref = mx.fast.rms_norm(norm_weight * experts, None, 1e-6)
        ref = (ref * weights[..., None]).sum(axis=-2)
        self._assert_close(out, ref, TOL[mx.float32])

    def test_rms_norm_sigmoid_gate(self):
        B, L, H, D = 2, 5, 3, 80
        for dtype in (mx.float32, mx.bfloat16):
            mx.random.seed(4)
            # Strided, as in the attention.
            x = mx.random.normal((B, L, H, D)).astype(dtype).transpose(0, 2, 1, 3)
            gate = 2 * mx.random.normal((B, L, 2 * H * D))
            gate = gate.astype(dtype)[..., H * D :]
            out = rms_norm_sigmoid_gate(x, gate, eps=1e-5)

            ref = mx.fast.rms_norm(x.astype(mx.float32), None, 1e-5)
            ref = ref.transpose(0, 2, 1, 3).reshape(B, L, H * D)
            ref = ref * mx.sigmoid(gate.astype(mx.float32))
            self._assert_close(out, ref.astype(dtype), TOL[dtype])


if __name__ == "__main__":
    unittest.main()
