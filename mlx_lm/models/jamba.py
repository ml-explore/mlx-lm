# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional, List, Any

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention, create_attention_mask
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: float
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attn_layer_offset: int
    attn_layer_period: int
    expert_layer_offset: int
    expert_layer_period: int
    mamba_d_conv: int
    mamba_d_state: int
    mamba_dt_rank: int
    mamba_expand: int
    num_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    max_position_embeddings: int
    vocab_size: int
    mamba_proj_bias: bool = False
    mamba_conv_bias: bool = True
    tie_word_embeddings: bool = True


class JambaMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class JambaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.args.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.args.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)