# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    attention_bias: bool
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    partial_rotary_factor: float
    rope_theta: float
    rope_traditional: bool = True
    max_position_embeddings: int = 32768


class Glm4MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            args.hidden_size, 2 * args.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, up_states = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up_states)


class Glm4Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        )
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            dims=int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=args.rope_traditional,
        )

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

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


class Glm4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Glm4Attention(args=args)

        self.mlp = Glm4MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_self_attn_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        x = x + self.post_self_attn_layernorm(
            self.self_attn(self.input_layernorm(x), mask, cache)
        )
        residual = x
        x = (
            self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(x)))
            + residual
        )
        return x


class Glm4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Glm4DecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, mask, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers
