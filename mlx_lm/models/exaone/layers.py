# Copyright © 2024 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.exaone.config import ModelArgs

__all__ = [
    "Attention",
    "AttentionModule",
    "MLP",
    "TransformerBlock",
]


class AttentionModule(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or (dim // n_heads)
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = AttentionModule(args)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.c_fc_0 = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.c_fc_1 = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(swiglu(self.c_fc_0(x), self.c_fc_1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.attn = Attention(args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attn.attention(self.ln_1(x), mask, cache)
        out = h + self.mlp(self.ln_2(h))
        return out
