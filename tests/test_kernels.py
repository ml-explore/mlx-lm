# Copyright © 2026 Apple Inc.

import unittest
from unittest.mock import patch

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
    sparse_mla,
    sparse_mla_attention,
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


def _sparse_mla_ref(q, q_pe, kv, k_pe, idx, scale):
    # Attention of each query over its selected rows; -1 entries are skipped.
    rows = mx.maximum(idx, 0)
    keys = mx.concatenate([kv, k_pe], axis=-1)[rows].astype(mx.float32)  # [L, K, r + p]
    queries = mx.concatenate([q, q_pe], axis=-1).astype(mx.float32)  # [H, L, r + p]
    scores = (queries[:, :, None, :] * keys[None]).sum(-1) * scale  # [H, L, K]
    scores = mx.where((idx >= 0)[None], scores, -mx.inf)
    probs = mx.softmax(scores, axis=-1)
    values = kv[rows].astype(mx.float32)  # [L, K, r]
    return (probs[..., None] * values[None]).sum(-2)  # [H, L, r]


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

    def test_sparse_mla_attention(self):
        # DSA sizes (512 latent + 64 rope), 16 heads, a top-k that isn't a multiple of
        # the kernel's key block, and skipped (-1) entries.
        H, L, N, K, r, p = 16, 7, 300, 100, 512, 64
        mx.random.seed(0)
        idx = mx.random.randint(0, N, (L, K)).astype(mx.int32)
        idx = mx.where(mx.random.uniform(shape=(L, K)) < 0.1, -1, idx)
        for dtype in (mx.bfloat16, mx.float16):
            q = mx.random.normal((H, L, r)).astype(dtype)
            q_pe = mx.random.normal((H, L, p)).astype(dtype)
            kv = mx.random.normal((N, r)).astype(dtype)
            k_pe = mx.random.normal((N, p)).astype(dtype)
            ref = _sparse_mla_ref(q, q_pe, kv, k_pe, idx, 0.05).astype(dtype)
            for variant in sparse_mla.VARIANTS:
                if not sparse_mla.is_available(variant):
                    continue
                out = sparse_mla_attention(
                    q, q_pe, kv, k_pe, idx, 0.05, variant=variant
                )
                self._assert_close(out, ref, 2e-2)

    def test_sparse_mla_tensor_ops_gate(self):
        # The condition of mlx's NAX kernels: macOS 26.2, generation 17 (18 for p GPUs).
        cases = [
            ("applegpu_g17d", (26, 2), True),
            ("applegpu_g17d", (26, 1), False),
            ("applegpu_g17p", (26, 2), False),
            ("applegpu_g18p", (26, 2), True),
            ("applegpu_g15d", (26, 2), False),
        ]
        for arch, version, expected in cases:
            with (
                patch.object(sparse_mla, "_gpu_architecture", return_value=arch),
                patch.object(sparse_mla, "_macos_version", return_value=version),
            ):
                self.assertEqual(sparse_mla.is_available("tensor_ops"), expected)

    def test_sparse_mla_unsupported_shapes(self):
        def inputs(H=16, r=512, p=16, dtype=mx.bfloat16):
            q, q_pe = mx.zeros((H, 2, r), dtype), mx.zeros((H, 2, p), dtype)
            kv, k_pe = mx.zeros((40, r), dtype), mx.zeros((40, p), dtype)
            return q, q_pe, kv, k_pe, mx.zeros((2, 4), mx.int32), 1.0

        # A latent width the kernel can't split across its simdgroups.
        self.assertIsNone(sparse_mla_attention(*inputs(r=32)))
        self.assertIsNone(sparse_mla_attention(*inputs(dtype=mx.float32)))
        self.assertIsNone(sparse_mla_attention(*inputs(H=8)))
        self.assertIsNone(sparse_mla_attention(*inputs(p=0)))
        self.assertIsNone(sparse_mla_attention(*inputs(), variant="unknown"))


if __name__ == "__main__":
    unittest.main()
