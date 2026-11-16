# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.spark2_5.config import ModelArgs

__all__ = [
    "Attention",
    "MLP",
    "TransformerBlock",
]


def _make_rope(args: ModelArgs, layer_type: str):
    params = args.rope_parameters[layer_type]
    base = float(params.get("rope_theta", 10000.0))
    factor = float(params.get("partial_rotary_factor", 1.0))
    dims = int(args.head_dim * factor)
    return nn.RoPE(dims, traditional=False, base=base)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        layer_type = args.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size

        self.q_k_v_proj = nn.Linear(
            args.hidden_size,
            q_size + 2 * kv_size,
            bias=args.attention_bias,
        )
        self.g_proj = nn.Linear(
            args.hidden_size,
            self.n_heads,
            bias=args.attention_bias,
        )
        self.out_proj = nn.Linear(
            q_size,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.rope = _make_rope(args, layer_type)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape

        qkv = self.q_k_v_proj(x)
        queries, keys, values = mx.split(
            qkv,
            [self.q_size, self.q_size + self.kv_size],
            axis=-1,
        )

        queries = queries.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # The Q/K/V views alias the fused projection. Materialize them before
        # RoPE so the rope kernel reads contiguous inputs on every backend.
        queries = mx.contiguous(queries)
        keys = mx.contiguous(keys)
        values = mx.contiguous(values)

        # The cache offset is the absolute token position, so RoPE stays
        # correct even after a rotating cache wraps its bounded storage.
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = mx.contiguous(output.transpose(0, 2, 1, 3))

        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = output * gate[..., None]

        return self.out_proj(output.reshape(batch_size, sequence_length, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.up_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.down_proj = nn.Linear(
            args.intermediate_size,
            args.hidden_size,
            bias=args.mlp_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    @property
    def is_sliding(self) -> bool:
        return self.self_attn.is_sliding

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attention_output = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden_states = x + attention_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + mlp_output
