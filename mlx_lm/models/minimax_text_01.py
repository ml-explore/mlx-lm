# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    attention_dropout: float
    attn_type_list: List[int]
    bos_token_id: Optional[int]
    eos_token_id: int
    head_dim: int
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layernorm_full_attention_alpha: float
    layernorm_full_attention_beta: float
    layernorm_linear_attention_alpha: float
    layernorm_linear_attention_beta: float
    layernorm_mlp_alpha: float
    layernorm_mlp_beta: float
    max_position_embeddings: int
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    num_local_experts: int
    output_router_logits: bool
    postnorm: bool
    rms_norm_eps: float
    rope_theta: int
    rotary_dim: int
    router_aux_loss_coef: float
    router_jitter_noise: float
    shared_intermediate_size: int
    sliding_window: Optional[int]
    tie_word_embeddings: bool
    vocab_size: int
    BLOCK: int = 256


class MiniMaxText01AttentionType0(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.num_heads = args.num_attention_heads
        self.offset = 0

        self.qkv_proj = nn.Linear(args.hidden_size, 3 * self.head_dim, bias=False)
        self.norm = nn.RMSNorm(args.hidden_size)
        self.output_gate = nn.Linear(args.hidden_size, self.head_dim * self.num_heads, bias=False)
        self.out_proj = nn.Linear(self.head_dim * self.num_heads, args.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return x


class MiniMaxText01AttentionType1(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads

        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = initialize_rope(
            dims=args.rotary_dim or self.head_dim,
            base=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

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
