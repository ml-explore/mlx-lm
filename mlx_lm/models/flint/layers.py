# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.kernels import expert_routing
from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.flint.config import ModelArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "MLP",
    "SparseMoeBlock",
]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((n_heads,))

        if args.layer_types[layer_idx] == "sliding_attention":
            self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.rope is not None:
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
            sinks=self.softmax_offset,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self._zero_bias = mx.zeros((args.num_experts,))

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        logits = self.gate(x.astype(mx.float32))

        k = self.top_k
        # The routing kernel has no vjp and needs Metal.
        if self.training or not mx.metal.is_available():
            scores = mx.sigmoid(logits)
            inds = mx.stop_gradient(
                mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            )
            weights = mx.take_along_axis(scores, inds, axis=-1)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        else:
            inds, weights = expert_routing(
                logits,
                self._zero_bias,
                top_k=k,
                multiplier=1.0,
                score_eps=0.0,
                route_scale=1.0,
            )

        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_expert(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, layer_idx)
        if layer_idx in args.mlp_only_layers:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))
