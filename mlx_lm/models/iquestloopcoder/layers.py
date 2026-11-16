# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.iquestloopcoder.config import ModelArgs

__all__ = [
    "Attention",
    "LoopGateProjection",
    "MLP",
    "TransformerBlock",
]


@partial(mx.compile, shapeless=True)
def _compute_gate(query: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    gate_logits = query @ weight[:, None, :].swapaxes(-1, -2)
    gate_logits = gate_logits + bias[..., None, None]
    return mx.sigmoid(gate_logits)


@partial(mx.compile, shapeless=True)
def _silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


class LoopGateProjection(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads,))

    def __call__(self, query: mx.array) -> mx.array:
        return _compute_gate(query, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def get_qkv(
        self, x: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        return queries, keys, values

    def attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu_mul(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
