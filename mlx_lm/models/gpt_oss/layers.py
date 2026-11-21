# Copyright © 2025 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import SwigluOAI
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.gpt_oss.config import ModelArgs

__all__ = [
    "AttentionBlock",
    "MLPBlock",
    "TransformerBlock",
    "mlx_topk",
]


class AttentionBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.sinks = mx.zeros((config.num_attention_heads,))

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )

        self.o_proj = nn.Linear(
            self.head_dim * config.num_attention_heads, config.hidden_size, bias=True
        )

        self.sm_scale = 1 / math.sqrt(config.head_dim)

        self.rope = initialize_rope(
            self.head_dim,
            config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
        )

    def __call__(self, x: mx.array, mask: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        D = self.head_dim

        q = self.q_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)
        k = self.k_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)
        v = self.v_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        v_hat = scaled_dot_product_attention(
            q, k, v, cache, self.sm_scale, mask=mask, sinks=self.sinks
        )

        return self.o_proj(v_hat.swapaxes(1, 2).reshape(B, L, -1))


# These operators emulate particular methods in torch that don't exist in MLX natively
def mlx_topk(a, k, axis=-1):
    """MLX equivalent of torch.topk"""
    partitioned_indices = mx.argpartition(a, kth=-k, axis=axis)
    # Extract only the top k indices (last k elements after partition)
    top_k_indices = mx.stop_gradient(partitioned_indices[..., -k:])
    # Get the corresponding values
    top_k_values = mx.take_along_axis(a, top_k_indices, axis=axis)
    return top_k_values, top_k_indices


class MLPBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_local_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
            num_experts=config.num_local_experts,
            activation=SwigluOAI(),
            bias=True,
        )
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        g = self.router(x)
        experts, indices = mlx_topk(g, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(experts, axis=-1, precise=True)

        # Experts block
        x = self.experts(x, indices)

        x = x * mx.expand_dims(expert_weights, axis=-1)

        y = x.sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = AttentionBlock(config)
        self.mlp = MLPBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: mx.array, cache=None) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x
