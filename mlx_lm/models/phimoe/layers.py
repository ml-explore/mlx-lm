# Copyright © 2024 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import SuScaledRoPE
from mlx_lm.models.phimoe.config import ModelArgs

__all__ = [
    "Attention",
    "PhiMoEDecoderLayer",
    "PhiMoESparseMoeBlock",
    "sparsemixer",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=True)

        # A null rope_scaling (e.g. Phi-mini-MoE) means plain RoPE.
        if args.rope_scaling is None:
            self.rope = nn.RoPE(head_dim, base=args.rope_theta)
        else:
            self.rope = SuScaledRoPE(
                head_dim,
                base=args.rope_theta,
                max_position_embeddings=args.max_position_embeddings,
                original_max_position_embeddings=args.original_max_position_embeddings,
                short_factor=args.rope_scaling["short_factor"],
                long_factor=args.rope_scaling["long_factor"],
                short_mscale=args.rope_scaling["short_mscale"],
                long_mscale=args.rope_scaling["long_mscale"],
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
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


def sparsemixer(
    gates: mx.array, *, top_k: int, jitter_eps: float
) -> tuple[mx.array, mx.array]:
    # Inference path of HF's sparsemixer.
    masked = gates
    inds, scores = [], []
    for _ in range(top_k):
        top = masked.max(axis=-1, keepdims=True)
        ind = mx.stop_gradient(mx.argmax(masked, axis=-1, keepdims=True))
        factor = mx.maximum(mx.abs(gates), top)
        band = mx.where((top - gates) / factor > 2 * jitter_eps, -mx.inf, masked)
        band = mx.softmax(band, axis=-1, precise=True)
        scores.append(mx.take_along_axis(band, ind, axis=-1))
        inds.append(ind)
        masked = mx.put_along_axis(
            masked, ind, mx.array(-mx.inf, masked.dtype), axis=-1
        )
    return mx.concatenate(inds, axis=-1), mx.concatenate(scores, axis=-1)


class PhiMoESparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size
        self.num_experts = args.num_local_experts
        self.top_k = args.num_experts_per_tok
        self.jitter_eps = args.router_jitter_noise
        if self.top_k > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.top_k}) is larger than "
                f"num_local_experts ({self.num_experts})."
            )

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(self.hidden_dim, self.ffn_dim, self.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)

        inds, scores = sparsemixer(gates, top_k=self.top_k, jitter_eps=self.jitter_eps)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y


class PhiMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = Attention(args)
        self.block_sparse_moe = PhiMoESparseMoeBlock(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        residual = x
        hidden_states = self.input_layernorm(x)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
