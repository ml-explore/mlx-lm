# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: Optional[int]
    num_key_value_heads: Optional[int]
    attention_bias: bool
    mlp_bias: bool
    rope_theta: float
    layer_types: List[str]
    sliding_window: int
    rope_traditional: bool = False
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True


class Olmo3Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim =  args.head_dim or args.hidden_size // n_heads
        self.layer_idx = layer_idx

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=args.attention_bias)

        self.q_norm = nn.RMSNorm(dims=self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=self.head_dim, eps=args.rms_norm_eps)
        self.is_sliding = (layer_idx + 1) % args.layer_types[layer_idx] != 0

        rope_base = args.rope_theta
        if self.is_sliding:
            self.rope = nn.RoPE(self.head_dim, traditional=args.rope_traditional, base=rope_base)
        else:
            self.rope = nn.RoPE(
                self.head_dim,
                traditional=args.rope_traditional,
                base=rope_base,
                scale=args.rope_scaling if hasattr(args, "rope_scaling") else 1.0
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if isinstance(mask, mx.array) and mask.shape[-1] != keys.shape[-2]:
            mask = mask[..., -keys.shape[-2] :]
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)