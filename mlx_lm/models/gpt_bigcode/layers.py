# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.gpt_bigcode.config import ModelArgs

__all__ = [
    "Attention",
    "MLP",
    "TransformerBlock",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = dim = args.n_embd
        self.n_heads = n_heads = args.n_head
        self.n_kv_heads = n_kv_heads = 1 if args.multi_query else args.n_head

        self.head_dim = head_dim = dim // n_heads

        self.kv_dim = n_kv_heads * head_dim

        self.scale = head_dim**-0.5

        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.c_attn = nn.Linear(dim, dim + 2 * self.kv_dim, bias=attention_bias)
        self.c_proj = nn.Linear(dim, dim, bias=attention_bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(
            qkv, [self.dim, self.dim + self.kv_dim], axis=-1
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.n_embd
        hidden_dim = args.n_inner
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.c_fc = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return out
