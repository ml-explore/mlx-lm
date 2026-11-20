# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.models.lfm2_moe.config import ModelArgs

__all__ = [
    "Attention",
    "Lfm2DecoderLayer",
    "Lfm2MoeSparseMoeBlock",
    "MLP",
    "ShortConv",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)

        if cache is not None:
            if cache[0] is None:
                state = mx.zeros(
                    (Bx.shape[0], self.L_cache - 1, self.args.hidden_size),
                    dtype=Bx.dtype,
                )
            else:
                state = cache[0]
            Bx = mx.concatenate([state, Bx], axis=1)
            n_keep = self.L_cache - 1
            t = x.shape[1]
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, t)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(Bx, positions, axis=1)
            else:
                cache[0] = Bx[:, -n_keep:, :]
            cache.advance(t)
        else:
            Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])

        conv_out = self.conv(Bx)

        y = C * conv_out
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_expert_bias = args.use_expert_bias
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)
        if self.use_expert_bias:
            self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(
        self,
        x: mx.array,
    ):
        # Sigmoid-gated routing (matches the HF Transformers Lfm2Moe reference):
        # the expert_bias is used only to select the top-k experts; the routing
        # weights are gathered from the unbiased sigmoid scores and then scaled by
        # routed_scaling_factor.
        routing_weights = mx.sigmoid(self.gate(x))

        k = self.top_k
        if self.use_expert_bias:
            scores_for_routing = routing_weights.astype(mx.float32) + self.expert_bias
            inds = mx.argpartition(scores_for_routing, kth=-k, axis=-1)[..., -k:]
        else:
            inds = mx.argpartition(routing_weights, kth=-k, axis=-1)[..., -k:]
        inds = mx.stop_gradient(inds)

        scores = mx.take_along_axis(routing_weights, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
        scores = scores * self.routed_scaling_factor
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = (
            MLP(
                config=args,
                intermediate_size=args.intermediate_size,
            )
            if layer_idx < args.num_dense_layers
            else Lfm2MoeSparseMoeBlock(args)
        )

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
