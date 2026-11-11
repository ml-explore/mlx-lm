# Copyright © 2026 Apple Inc.

import itertools
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.kernels import (
    can_combine,
    expert_routing,
    norm_add_norm,
    post_expert_norm_combine,
    pre_norm,
    qk_norm,
    qk_norm_rope,
    rms_norm_sigmoid_gate,
    sparse_mla,
    sparse_mla_attention,
    switch_glu_combine,
)
from mlx_lm.models import (
    Klear,
    afmoe,
    bailing_hybrid,
    bailing_moe,
    bailing_moe_linear,
    bailing_moe_v3,
    deepseek,
    deepseek_v2,
    deepseek_v3,
    deepseek_v32,
    dots1,
    exaone_moe,
    gemma4_text,
    glm4_moe,
    glm4_moe_lite,
    gpt_oss,
    granitemoehybrid,
    hunyuan,
    jamba,
    kimi_k3,
    laguna,
    lfm2_moe,
    longcat_flash,
    mellum,
    mimo_v2_flash,
    minimax,
    minimax_m3_vl,
    mixtral,
    olmoe,
    qwen2_moe,
    qwen3_moe,
    qwen3_next,
    step3p5,
    switch_layers,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

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


def _switch_glu(dims, hidden, experts, *, bits, group_size, dtype, activation=None):
    mx.random.seed(4)
    # Not `activation or ...`: a module without parameters is an empty dict.
    if activation is None:
        activation = SwiGLU()
    switch_glu = SwitchGLU(dims, hidden, experts, activation=activation)
    switch_glu.set_dtype(dtype)
    nn.quantize(switch_glu, group_size=group_size, bits=bits)
    switch_glu.eval()
    return switch_glu


def _switch_glu_combine_ref(switch_glu, x, inds, weights, act=nn.silu):
    # The selected experts in fp32, from the dequantized weights.
    def expert(proj, e):
        w = mx.dequantize(
            proj.weight[e], proj.scales[e], proj.biases[e], proj.group_size, proj.bits
        )
        return w.astype(mx.float32)

    x = x.reshape(-1).astype(mx.float32)
    y = mx.zeros_like(x)
    for e, w in zip(inds.reshape(-1).tolist(), weights.reshape(-1).tolist()):
        gate = expert(switch_glu.gate_proj, e) @ x
        up = expert(switch_glu.up_proj, e) @ x
        y = y + w * (expert(switch_glu.down_proj, e) @ (act(gate) * up))
    return y


# One config for the MoE blocks below: each reads the fields it needs.
# One config for the MoE blocks below: each reads the fields it needs.
_MOE_ARGS = SimpleNamespace(
    hidden_size=128,
    intermediate_size=64,
    moe_intermediate_size=64,
    num_experts=16,
    num_local_experts=16,
    n_routed_experts=16,
    num_experts_per_tok=4,
    top_k_experts=4,
    norm_topk_prob=True,
    shared_expert_intermediate_size=64,
    n_shared_experts=1,
    num_shared_experts=1,
    mlp_bias=False,
    rms_norm_eps=1e-6,
    routed_scaling_factor=2.5,
    use_expert_bias=True,
    n_group=4,
    topk_group=2,
    topk_method="noaux_tc",
    score_func="sigmoid",
    route_norm=True,
    route_scale=2.5,
    moe_topk=4,
    num_shared_expert=1,
    use_mixed_mlp_moe=True,
    moe_num_experts=16,
    moe_top_k=4,
    moe_router_scaling_factor=2.5,
    norm_expert_weight=True,
    share_expert_dim=64,
    swiglu_limits=None,
    swiglu_limits_shared=None,
    shared_intermediate_size=64,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    moe_routed_scaling_factor=2.5,
    moe_router_logit_softcapping=0.0,
    moe_router_score_func="sigmoid",
    score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_enable_routed_scaling=True,
    moe_router_enable_shared_expert=True,
    moe_shared_expert_intermediate_size=64,
    use_bias=False,
    num_experts_per_token=4,
    num_expert_group=4,
    moe_renormalize=True,
    routed_expert_hidden_size=None,
    latent_moe_use_norm=False,
    activation_situ_beta=1.0,
    activation_situ_linear_beta=1.0,
    zero_expert_num=2,
    zero_expert_type="identity",
    expert_ffn_hidden_size=64,
    router_bias=False,
)


class _Gemma4MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.router = gemma4_text.Router(args)
        self.experts = gemma4_text.Experts(args)

    def __call__(self, x):
        return self.experts(x, *self.router(x))


def _tiny_qwen3_moe():
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        num_experts=16,
        num_experts_per_tok=4,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        norm_topk_prob=True,
    )
    mx.random.seed(6)
    model = qwen3_moe.Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    model.eval()
    return model


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

    def test_switch_glu_combine(self):
        # 640 values per row leave the last block of 512 partial; 384 is not a
        # multiple of 512 either, so lanes straddle two experts' rows of the down
        # projection.
        dims, hidden, experts = 640, 384, 16
        inds = mx.array([[[3, 11, 0, 7]]], dtype=mx.uint32)
        weights = mx.array([[[0.4, 0.3, 0.2, 0.1]]])
        activations = ((SwiGLU(), nn.silu), (gemma4_text.GeGLU(), nn.gelu_approx))
        for bits, group_size, dtype, (activation, act) in itertools.product(
            (2, 3, 4, 5, 6, 8), (32, 64, 128), (mx.bfloat16, mx.float16), activations
        ):
            switch_glu = _switch_glu(
                dims,
                hidden,
                experts,
                bits=bits,
                group_size=group_size,
                dtype=dtype,
                activation=activation,
            )
            x = mx.random.normal((1, 1, dims)).astype(dtype)
            self.assertTrue(can_combine(switch_glu, x))
            ref = _switch_glu_combine_ref(switch_glu, x, inds, weights, act)
            ref = ref.reshape(x.shape)
            # The output has the dtype of the weighted sum: float32 weights promote.
            for w in (weights, weights.astype(dtype)):
                out = switch_glu_combine(switch_glu, x, inds, w)
                ref_dtype = mx.result_type(dtype, w.dtype)
                self._assert_close(out, ref.astype(ref_dtype), TOL[mx.bfloat16])

    def test_can_combine(self):
        x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
        switch_glu = _switch_glu(512, 256, 8, bits=4, group_size=64, dtype=mx.bfloat16)
        self.assertTrue(can_combine(switch_glu, x))
        self.assertFalse(can_combine(switch_glu, mx.concatenate([x, x], axis=1)))
        self.assertFalse(can_combine(switch_glu, x.astype(mx.float32)))
        with mx.stream(mx.cpu):
            self.assertFalse(can_combine(switch_glu, x))
        switch_glu.train()
        self.assertFalse(can_combine(switch_glu, x))
        for bits, dtype in ((1, mx.bfloat16), (4, mx.float32)):
            other = _switch_glu(512, 256, 8, bits=bits, group_size=64, dtype=dtype)
            self.assertFalse(can_combine(other, x))
        unquantized = SwitchGLU(512, 256, 8)
        unquantized.set_dtype(mx.bfloat16)
        self.assertFalse(can_combine(unquantized, x))

    def test_qwen3_moe_decode_combine(self):
        # A decode step with the fused experts against the generic path, which
        # rounds to bf16 in other places.
        model = _tiny_qwen3_moe()
        prompt = mx.random.randint(0, 1000, (1, 8))

        def decode():
            cache = make_prompt_cache(model)
            model(prompt, cache=cache)
            return model(mx.array([[5]]), cache=cache)

        out = decode()
        with patch.object(switch_layers, "can_combine", lambda *args: False):
            ref = decode()
        self._assert_close(out, ref, 3e-2)
        self.assertTrue(mx.array_equal(out.argmax(-1), ref.argmax(-1)))

    def test_moe_blocks_combine(self):
        # Each model's MoE block on one token with SwitchGLU.combine in the fused
        # kernels, against the generic path.
        greedy = SimpleNamespace(**{**vars(_MOE_ARGS), "topk_method": "greedy"})
        blocks = [
            (qwen3_next, qwen3_next.Qwen3NextSparseMoeBlock, _MOE_ARGS),
            (qwen2_moe, qwen2_moe.Qwen2MoeSparseMoeBlock, _MOE_ARGS),
            (mixtral, mixtral.MixtralSparseMoeBlock, _MOE_ARGS),
            (olmoe, olmoe.OlmoeSparseMoeBlock, _MOE_ARGS),
            (gemma4_text, _Gemma4MoE, _MOE_ARGS),
            (deepseek_v3, deepseek_v3.DeepseekV3MoE, _MOE_ARGS),
            (deepseek_v32, deepseek_v32.DeepseekV32MoE, _MOE_ARGS),
            (glm4_moe, glm4_moe.MoE, _MOE_ARGS),
            (glm4_moe_lite, glm4_moe_lite.Glm4MoeLiteMoE, _MOE_ARGS),
            (exaone_moe, exaone_moe.MoE, _MOE_ARGS),
            (mimo_v2_flash, mimo_v2_flash.MoE, _MOE_ARGS),
            (minimax, minimax.MiniMaxSparseMoeBlock, _MOE_ARGS),
            (lfm2_moe, lfm2_moe.Lfm2MoeSparseMoeBlock, _MOE_ARGS),
            (granitemoehybrid, granitemoehybrid.GraniteMoeHybridMoE, _MOE_ARGS),
            (jamba, jamba.JambaSparseMoeBlock, _MOE_ARGS),
            (hunyuan, hunyuan.MoeBlock, _MOE_ARGS),
            (mellum, mellum.MellumSparseMoeBlock, _MOE_ARGS),
            (deepseek, deepseek.DeepseekMoE, _MOE_ARGS),
            (deepseek_v2, deepseek_v2.DeepseekV2MoE, greedy),
            (afmoe, afmoe.AfmoeMoE, _MOE_ARGS),
            (dots1, dots1.Dots1MoE, _MOE_ARGS),
            (step3p5, partial(step3p5.Step3p5MoE, layer_idx=0), _MOE_ARGS),
            (Klear, Klear.KlearSparseMoeBlock, _MOE_ARGS),
            (minimax_m3_vl, minimax_m3_vl.SparseMoeBlock, _MOE_ARGS),
            (laguna, laguna.MoE, _MOE_ARGS),
            (bailing_moe, bailing_moe.BailingMoeSparseMoeBlock, _MOE_ARGS),
            (bailing_moe_linear, bailing_moe_linear.SparseMoeBlock, _MOE_ARGS),
            (bailing_hybrid, bailing_hybrid.SparseMoeBlock, _MOE_ARGS),
            (bailing_moe_v3, bailing_moe_v3.BailingSparseMoE, _MOE_ARGS),
            (kimi_k3, kimi_k3.KimiK3SparseMoE, _MOE_ARGS),
            (gpt_oss, gpt_oss.MLPBlock, _MOE_ARGS),
            (longcat_flash, longcat_flash.LongcatFlashMoE, _MOE_ARGS),
        ]
        # SwigluOAI and SiTU have no Metal kernel, nor experts with biases.
        unfused = {minimax_m3_vl, kimi_k3, gpt_oss}
        for module, block_cls, args in blocks:
            with self.subTest(module.__name__):
                mx.random.seed(7)
                block = block_cls(args)
                block.set_dtype(mx.bfloat16)
                nn.quantize(block, group_size=64, bits=4)
                block.eval()
                x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
                with patch.object(
                    switch_layers,
                    "switch_glu_combine",
                    wraps=switch_layers.switch_glu_combine,
                ) as fused:
                    out = block(x)
                self.assertEqual(fused.call_count, 0 if module in unfused else 1)
                with patch.object(switch_layers, "can_combine", lambda *args: False):
                    ref = block(x)
                self._assert_close(out, ref, 3e-2)

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
