# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.models.gpt_neox.config import ModelArgs

__all__ = [
    "Attention",
    "MLP",
    "TransformerBlock",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert (
            args.hidden_size % args.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        self.hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.rope = nn.RoPE(
            dims=int(self.head_dim * args.rotary_pct),
            traditional=False,
            base=args.rotary_emb_base,
        )

        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)

        new_qkv_shape = qkv.shape[:-1] + (self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.reshape(*new_qkv_shape)

        queries, keys, values = [x.transpose(0, 2, 1, 3) for x in qkv.split(3, -1)]

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
        return self.dense(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.dense_h_to_4h = nn.Linear(self.hidden_size, args.intermediate_size)
        self.dense_4h_to_h = nn.Linear(args.intermediate_size, self.hidden_size)

    def __call__(self, x) -> mx.array:
        # gelu_approx corresponds to FastGELUActivation in transformers.
        return self.dense_4h_to_h(nn.gelu_approx(self.dense_h_to_4h(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.layer_norm_eps = args.layer_norm_eps
        self.attention = Attention(args)
        self.mlp = MLP(args)
        self.use_parallel_residual = args.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            self.hidden_size,
            eps=self.layer_norm_eps,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.use_parallel_residual:
            residual = x
            # Run attention and feedforward network in parallel.
            attn = self.attention(self.input_layernorm(x), mask, cache)
            ffn = self.mlp(self.post_attention_layernorm(x))
            out = attn + ffn + residual
            return out
        else:
            # Run attention and feedforward network sequentially.
            attn_output = self.attention(self.input_layernorm(x), mask, cache)
            x = x + attn_output
            ffn_output = self.mlp(self.post_attention_layernorm(x))
            x = x + ffn_output
            return x
