# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    QuantizedAllToShardedLinear,
    QuantizedShardedToAllLinear,
)

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.ministral3.config import ModelArgs

__all__ = [
    "Attention",
    "Fp8AllToShardedLinear",
    "Fp8Linear",
    "Fp8ShardedToAllLinear",
    "FusedGLU",
    "MLP",
    "TransformerBlock",
]


class Fp8Linear(nn.QuantizedLinear):
    """Linear on e4m3 weights, run as mxfp8 with unit group scales.

    The kernel has no slot for the per-tensor scale, so it scales the output.
    Fused layers have one scale per output row.
    """

    def __init__(self, input_dims: int, output_dims: int, per_row: bool = False):
        super().__init__(input_dims, output_dims, False, 32, 8, "mxfp8")
        self.output_scale = mx.ones((output_dims,)) if per_row else mx.array(1.0)
        self.freeze()

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class Fp8AllToShardedLinear(QuantizedAllToShardedLinear):
    """Fp8Linear with the outputs sharded across the group."""

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class Fp8ShardedToAllLinear(QuantizedShardedToAllLinear):
    """Fp8Linear with the inputs sharded across the group."""

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class FusedGLU(nn.Module):
    """Gated MLP with gate_proj and up_proj fused into one projection."""

    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dims, 2 * hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, input_dims, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


def _get_llama_4_attn_scale(size, offset, beta: float, max_position_embeddings: int):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]

    scaling = 1 + beta * mx.log(
        1 + mx.floor((mx.arange(size) + offset) / max_position_embeddings)
    )
    if scaling.ndim == 2:
        return scaling[:, None, :, None]
    else:
        return scaling[:, None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        self.scaling_beta = args.rope_parameters["llama_4_scaling_beta"]
        self.original_max_position_embeddings = args.rope_parameters[
            "original_max_position_embeddings"
        ]

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_parameters["rope_theta"],
            False,
            args.rope_parameters,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # sanitize fuses the three projections unless they are stored differently.
        if "qkv_proj" in self:
            q, kv = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            queries, keys, values = mx.split(self.qkv_proj(x), [q, q + kv], axis=-1)
        else:
            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        if attn_scale is None:
            attn_scale = _get_llama_4_attn_scale(
                L,
                offset,
                self.scaling_beta,
                self.original_max_position_embeddings,
            ).astype(x.dtype)
        queries = queries * attn_scale
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
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        attn_scale: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
