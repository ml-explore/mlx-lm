# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.Klear.config import ModelArgs

__all__ = [
    "KlearAttention",
    "KlearDecoderLayer",
    "KlearMLP",
    "KlearSparseMoeBlock",
]


class KlearAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads

        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(B, L, self.num_attention_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class KlearMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class KlearSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = KlearMLP(
            args.hidden_size,
            hidden_dim=args.moe_intermediate_size * args.n_shared_experts,
        )
        self.coefficient = nn.Linear(args.hidden_size, 2)
        self.expert_bias = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        routing_weights = mx.sigmoid(self.gate(x).astype(mx.float32))
        biased_weights = routing_weights + self.expert_bias.reshape((1, 1, -1))
        k = self.top_k
        inds = mx.argpartition(-biased_weights, kth=k - 1, axis=-1)[..., :k]
        inds = mx.stop_gradient(inds)
        scores = mx.take_along_axis(routing_weights, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        expert_out = self.experts(x, inds)
        y_experts = (expert_out * scores[..., None]).sum(axis=-2)
        coef = mx.softmax(self.coefficient(x), axis=-1, precise=True)
        shared = self.shared_experts(x)
        y = y_experts * coef[..., :1] + shared * coef[..., 1:]
        return y


class KlearDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = KlearAttention(args)

        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = KlearSparseMoeBlock(args)
        else:
            self.mlp = KlearMLP(args.hidden_size, args.intermediate_size)

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
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
