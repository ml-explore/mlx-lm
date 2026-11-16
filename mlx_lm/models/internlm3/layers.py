# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.internlm3.config import ModelArgs

__all__ = [
    "Attention",
    "DynamicNTKScalingRoPE",
    "MLP",
    "TransformerBlock",
]


class DynamicNTKScalingRoPE(nn.Module):
    """Implements the rotary positional encoding with Dynamic NTK scaling."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.original_base = base
        self.dims = dims
        self.traditional = traditional
        self.scale = scale

    def extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}, max_position_embeddings={self.max_position_embeddings}, scaling_factor={self.scaling_factor}"

    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        if seq_len > self.max_position_embeddings:
            base = self.original_base * (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.dims / (self.dims - 2))
        else:
            base = self.original_base

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        qkv_bias = args.qkv_bias
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.n_kv_groups = n_heads // args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=qkv_bias)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, args.bias)
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
