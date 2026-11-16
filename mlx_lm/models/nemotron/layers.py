# Copyright © 2024 Apple Inc.

from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.nemotron.config import ModelArgs

__all__ = [
    "Attention",
    "MLP",
    "NemotronLayerNorm1P",
    "TransformerBlock",
    "relu_squared",
]


@partial(mx.compile, shapeless=True)
def relu_squared(x):
    return nn.relu(x).square()


class NemotronLayerNorm1P(nn.LayerNorm):
    def __call__(self, x):
        weight = self.weight + 1 if "weight" in self else None
        bias = self.bias if "bias" in self else None
        return mx.fast.layer_norm(x, weight, bias, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.partial_rotary_factor = args.partial_rotary_factor

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        rope_scale = 1.0
        if args.rope_scaling and args.rope_scaling["type"] == "linear":
            assert isinstance(args.rope_scaling["factor"], float)
            rope_scale = 1 / args.rope_scaling["factor"]
        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

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
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        mlp_bias = args.mlp_bias

        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(relu_squared(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = NemotronLayerNorm1P(args.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = NemotronLayerNorm1P(
            args.hidden_size, eps=args.norm_eps
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
