# Copyright © 2025 Apple Inc.

from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import MultiLinear, scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.recurrent import gated_delta_update, normalize_qk
from mlx_lm.models.cache import KVCache
from mlx_lm.models.kimi_linear.config import ModelArgs

__all__ = [
    "KimiDecoderLayer",
    "KimiDeltaAttention",
    "KimiMLAAttention",
    "KimiMLP",
    "KimiSparseMoE",
    "ShortConv1d",
]


class KimiMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    elif score_function == "softmax":
        scores = mx.softmax(gates, axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class KimiSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        if experts is None:
            raise ValueError("num_experts must be specified for MoE layers")

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, args.moe_intermediate_size, experts)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        out = self.switch_mlp(x, inds)
        out = (out * weights[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class KimiMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

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
        return self.o_proj(output)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])

        return out, new_state


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.expand_dims(
            mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))),
            (0, 1, 3),
        )
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = None
            k_state = None
            v_state = None
            ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = s
            k_state = s
            v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        q, k = normalize_qk(q, k, inv_scale=self.scale, eps=1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiSparseMoE(args)
        else:
            self.mlp = KimiMLP(args)

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
        attn_cache = None if cache is None else cache
        y = self.self_attn(self.input_layernorm(x), mask, attn_cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z
