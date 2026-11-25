# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import MultiLinear, scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.recurrent import gated_delta_update
from mlx_lm.models.omoikane.config import ModelArgs
from mlx_lm.models.qwen3_next.layers import Qwen3NextMLP as MLP
from mlx_lm.models.qwen3_next.layers import Qwen3NextRMSNormGated as RMSNormGated

__all__ = [
    "OmoikaneAttention",
    "OmoikaneDecoderLayer",
    "OmoikaneGatedDeltaNet",
    "OmoikaneMoE",
    "OmoikaneMoEGate",
    "group_limited_select",
    "l2_normalize",
]


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    """``x / sqrt(sum(x^2) + eps)`` over the last axis.

    Unlike ``normalize_qk``, this folds no ``head_dim**-0.5`` readout scale
    into the queries: Omoikane reads the state out with a scale of 1.
    """
    d = x.shape[-1]
    return d**-0.5 * mx.fast.rms_norm(x, None, eps / d)


class OmoikaneGatedDeltaNet(nn.Module):
    """Gated delta rule with a short causal convolution on q, k and v.

    The per-head output is RMS-normalised and gated by ``silu(g_proj(x))``.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        cfg = args.gdn_config
        self.num_heads = cfg["num_heads"]
        self.head_k_dim = cfg["head_dim_k"]
        self.head_v_dim = cfg["head_dim_v"]
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.conv_kernel_size = cfg["short_conv_kernel"]
        self.qk_l2norm = cfg["qk_l2norm"]
        dim = args.hidden_size

        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.a_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

        def conv(channels):
            return nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=self.conv_kernel_size,
                groups=channels,
                padding=0,
                bias=cfg["conv_bias"],
            )

        self.q_conv1d = conv(self.key_dim)
        self.k_conv1d = conv(self.key_dim)
        self.v_conv1d = conv(self.value_dim)

        self.A_log = mx.zeros(self.num_heads)
        self.dt_bias = mx.zeros(self.num_heads)
        self.o_norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        H, Dk, Dv = self.num_heads, self.head_k_dim, self.head_v_dim

        qkv = mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], -1)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        n_keep = self.conv_kernel_size - 1
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, n_keep, qkv.shape[-1]), dtype=qkv.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)

        if cache is not None:
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        # As in training, the delta rule runs in float32 whatever the model's
        # dtype: q, k and v after the convolution, the gates and the output.
        q, k, v = mx.split(conv_input, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = nn.silu(self.q_conv1d(q)).reshape(B, S, H, Dk).astype(mx.float32)
        k = nn.silu(self.k_conv1d(k)).reshape(B, S, H, Dk).astype(mx.float32)
        v = nn.silu(self.v_conv1d(v)).reshape(B, S, H, Dv).astype(mx.float32)
        if self.qk_l2norm:
            q, k = l2_normalize(q), l2_normalize(k)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            self.a_proj(x).astype(mx.float32),
            self.b_proj(x).astype(mx.float32),
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.o_norm(out, self.g_proj(x).reshape(B, S, H, Dv))
        return self.o_proj(out.reshape(B, S, -1).astype(x.dtype))


class OmoikaneAttention(nn.Module):
    """Multi-head latent attention with decoupled RoPE and a sigmoid output gate.

    The cache holds the normalised latent and the shared RoPE key, as in
    DeepSeek-V3. Unlike DeepSeek-V3, the RoPE dimensions are rotated in halves
    (not interleaved pairs), and the attention output is multiplied by
    ``sigmoid(o_gate_proj(x))`` before ``o_proj``.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        cfg = args.mla_config
        self.num_heads = cfg["num_heads"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.v_head_dim = cfg["v_head_dim"]
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5
        dim = args.hidden_size

        self.q_a_proj = nn.Linear(dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_gate_proj = nn.Linear(dim, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.qk_rope_head_dim, traditional=False, base=cfg["rope_theta"]
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(self.o_gate_proj(x)))


@mx.compile
def group_limited_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    """Sigmoid top-k routing within the best groups of experts.

    A group scores the sum of its two best corrected scores; experts outside
    the ``topk_group`` best groups are excluded with ``-inf`` (DeepSeek-V3's
    ``group_expert_select`` zeroes them instead, which differs once the
    correction bias makes scores negative). The weights are the uncorrected
    scores of the chosen experts.
    """
    scores = mx.sigmoid(gates.astype(mx.float32))
    corrected = scores + e_score_correction_bias
    if n_group > 1:
        corrected = mx.unflatten(corrected, axis=-1, shape=(n_group, -1))
        n_best = min(2, corrected.shape[-1])
        group_scores = mx.topk(corrected, n_best, axis=-1).sum(axis=-1, keepdims=True)
        kth = n_group - topk_group
        threshold = mx.sort(group_scores, axis=-2)[..., kth : kth + 1, :]
        corrected = mx.where(group_scores >= threshold, corrected, -mx.inf)
        corrected = mx.flatten(corrected, -2, -1)

    inds = mx.argpartition(-corrected, kth=top_k - 1, axis=-1)[..., :top_k]
    inds = mx.stop_gradient(inds)
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


class OmoikaneMoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        cfg = args.moe_config
        self.top_k = cfg["num_experts_per_tok"]
        self.n_group = cfg["group_limited_routing"]["n_group"]
        self.topk_group = cfg["group_limited_routing"]["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        n_experts = cfg["n_routed_experts"]
        self.weight = mx.zeros((n_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((n_experts,))

    def __call__(self, x: mx.array):
        # The router scores in float32 whatever the model's dtype.
        gates = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_limited_select(
            gates,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class OmoikaneMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        cfg = args.moe_config
        self.gate = OmoikaneMoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, cfg["moe_intermediate_size"], cfg["n_routed_experts"]
        )
        self.n_shared_experts = cfg["n_shared_experts"]
        if self.n_shared_experts:
            self.shared_experts = MLP(
                args.hidden_size, cfg["moe_intermediate_size"] * self.n_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.n_shared_experts:
            y = y + self.shared_experts(x)
        return y


class OmoikaneDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = layer_idx not in args.mla_layer_indices
        if self.is_linear:
            self.self_attn = OmoikaneGatedDeltaNet(args)
        else:
            self.self_attn = OmoikaneAttention(args)

        if layer_idx in args.dense_layer_indices or not args.moe_config:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = OmoikaneMoE(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))
