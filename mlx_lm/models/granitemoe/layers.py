# Copyright © 2023 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.granitemoe.config import ModelArgs

__all__ = [
    "GraniteMoeAttention",
    "GraniteMoeDecoderLayer",
    "GraniteMoeMoE",
    "GraniteMoeTopKGating",
]


class GraniteMoeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = args.attention_multiplier
        attention_bias = args.attention_bias
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
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


class GraniteMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array):
        logits = self.layer(hidden_states)
        top_k_idx = mx.stop_gradient(
            mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        )
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1)
        return top_k_idx, top_k_gates


class GraniteMoeMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.input_size = args.hidden_size
        self.hidden_size = args.intermediate_size
        self.switch_mlp = SwitchGLU(
            self.input_size, self.hidden_size, args.num_local_experts
        )
        self.router = GraniteMoeTopKGating(
            input_size=self.input_size,
            num_experts=args.num_local_experts,
            top_k=args.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        token_ids, gates = self.router(x)
        y = self.switch_mlp(x, token_ids)
        return (y * gates[..., None]).sum(axis=-2).astype(y.dtype)


class GraniteMoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = GraniteMoeAttention(args)
        self.block_sparse_moe = GraniteMoeMoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.residual_multiplier = args.residual_multiplier

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * self.residual_multiplier
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h + r * self.residual_multiplier
        return out
