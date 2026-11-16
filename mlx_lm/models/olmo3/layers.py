# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.olmo3.config import ModelArgs

__all__ = [
    "Olmo3Attention",
    "Olmo3DecoderLayer",
    "Olmo3MLP",
]


class Olmo3Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.layer_idx = layer_idx

        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
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
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(
            args.num_attention_heads * self.head_dim, eps=args.rms_norm_eps
        )
        self.k_norm = nn.RMSNorm(
            args.num_key_value_heads * self.head_dim, eps=args.rms_norm_eps
        )

        if args.layer_types[layer_idx] != "full_attention":
            self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)
        else:
            self.rope = initialize_rope(
                self.head_dim,
                traditional=False,
                base=args.rope_theta,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_norm(self.q_proj(x))
        keys = self.k_norm(self.k_proj(x))
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

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


class Olmo3MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Olmo3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Olmo3Attention(args, layer_idx=layer_idx)
        self.mlp = Olmo3MLP(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.post_attention_layernorm(self.self_attn(x, mask, cache))
        h = x + r
        r = self.post_feedforward_layernorm(self.mlp(h))
        out = h + r
        return out
