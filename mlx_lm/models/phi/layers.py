# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.phi.config import ModelArgs

__all__ = [
    "PhiAttention",
    "PhiDecoderLayer",
    "PhiMLP",
]


class PhiAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )

        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        B, L, D = queries.shape
        n_heads, n_kv_heads = self.num_heads, self.num_key_value_heads

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(
            B,
            L,
            n_heads,
            -1,
        ).moveaxis(1, 2)
        keys = keys.reshape(B, L, n_kv_heads, -1).moveaxis(1, 2)
        values = values.reshape(B, L, n_kv_heads, -1).moveaxis(1, 2)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scale = math.sqrt(1 / queries.shape[-1])
        output = scaled_dot_product_attention(
            queries.astype(mx.float32),
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        ).astype(values.dtype)

        output = output.moveaxis(2, 1).reshape(B, L, -1)

        return self.dense(output)


class PhiMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = PhiAttention(config=config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = PhiMLP(config)

    def __call__(self, x, mask, cache):
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x
