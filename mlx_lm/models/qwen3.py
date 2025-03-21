# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_dropout: float
    head_dim: int
    sliding_window: int
    max_window_layers: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_sliding_window: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False



class Qwen3MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
    

class Qwen3Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.use_sliding_window = args.use_sliding_window
        self.sliding_window = args.sliding_window

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(args.hidden_size, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, args.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

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
        cache: Optional[Any] = None
    ) -> mx.array:
        input_shape = x.shape[:-1]  # Gets all dimensions except the last one
        hidden_shape = (*input_shape, -1, self.head_dim)  # Shape for heads and head_dim

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape and apply normalization - matching the PyTorch implementation
        queries = self.q_norm(queries.reshape(hidden_shape)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(hidden_shape[:2] + (self.n_kv_heads, self.head_dim))).transpose(0, 2, 1, 3)
        values = values.reshape(hidden_shape[:2] + (self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        sliding_window = None if not self.use_sliding_window else self.sliding_window

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask, sliding_window=sliding_window
        )
        output = output.transpose(0, 2, 1, 3).reshape(*input_shape, -1)
        return self.o_proj(output)
    

class Qwen3Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Qwen3Attention(args)
        self.mlp = Qwen3MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
    

class Qwen3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3Block(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)