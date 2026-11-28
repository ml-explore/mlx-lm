# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.gpt2.config import ModelArgs

__all__ = [
    "Attention",
    "MLP",
    "TransformerBlock",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.head_dim = self.n_embd // self.n_head

        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

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

        self.n_embd = args.n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd)
        self.c_proj = nn.Linear(4 * self.n_embd, self.n_embd)

    def __call__(self, x) -> mx.array:
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.layer_norm_epsilon = args.layer_norm_epsilon
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(
            self.n_embd,
            eps=self.layer_norm_epsilon,
        )
        self.ln_2 = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

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
