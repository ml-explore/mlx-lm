# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gpt_oss"
    num_hidden_layers: int = 36
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-05
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rope_theta: int = 150000
    rope_scaling: Any = None
    layer_types: list = None


# These operators emulate particular methods in torch that don't exist in MLX natively
def mlx_topk(a, k, axis=-1):
    """MLX equivalent of torch.topk"""
    partitioned_indices = mx.argpartition(a, kth=-k, axis=axis)
    # Extract only the top k indices (last k elements after partition)
    top_k_indices = partitioned_indices[..., -k:]
    # Get the corresponding values
    top_k_values = mx.take_along_axis(a, top_k_indices, axis=axis)
    return top_k_values, top_k_indices


@partial(mx.compile, shapeless=True)
def swiglu(x_linear, x_glu, alpha: float = 1.702, limit: float = 7.0):
    # Clamp the input values
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)

    glu_scaled = alpha * x_glu
    sig = mx.sigmoid(glu_scaled)

    out_glu = x_glu * sig
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(x, gate)


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
        Hk = self.num_key_value_heads

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
            activation=SwiGLU(),
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


class GptOssMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layer_types = args.layer_types or [
            "sliding_attention",
            "full_attention",
        ] * (args.num_hidden_layers // 2)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.window_size = args.sliding_window
        self.swa_idx = self.layer_types.index("sliding_attention")
        self.ga_idx = self.layer_types.index("full_attention")

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            x = input_embeddings
        else:
            x = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(x, cache[self.ga_idx])
        swa_mask = create_attention_mask(
            x, cache[self.swa_idx], window_size=self.window_size
        )

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            x = layer(x, mask, c)
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GptOssMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights):
        if any("gate_proj.weight" in k for k in weights.keys()):
            return weights  # already sanitized

        new_weights = {}
        for k, v in weights.items():
            if "gate_up_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.view(mx.uint32).flatten(-2)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_weights[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(
                    v[..., ::2, :]
                )
                new_weights[k.replace("gate_up_proj", "up_proj")] = mx.contiguous(
                    v[..., 1::2, :]
                )
            elif "down_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.view(mx.uint32).flatten(-2)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_weights[k] = v
            elif "gate_up_proj_bias" in k:
                new_weights[k.replace("gate_up_proj_bias", "gate_proj.bias")] = (
                    mx.contiguous(v[..., ::2])
                )
                new_weights[k.replace("gate_up_proj_bias", "up_proj.bias")] = (
                    mx.contiguous(v[..., 1::2])
                )
            elif "down_proj_bias" in k:
                new_weights[k.replace("down_proj_bias", "down_proj.bias")] = v
            else:
                new_weights[k] = v

        return new_weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()

        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, sharding="all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, sharding="all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, sharding="all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, sharding="sharded-to-all", group=group
            )

            layer.self_attn.num_attention_heads //= N
            layer.self_attn.num_key_value_heads //= N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads
                * R : layer.self_attn.num_attention_heads
                * (R + 1)
            ]

            shard_inplace(layer.mlp.experts.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.experts.down_proj, "sharded-to-all", group=group)
            layer.mlp.experts.down_proj.bias /= N
            shard_inplace(
                layer.mlp.experts.up_proj, sharding="all-to-sharded", group=group
            )

            layer.mlp.sharding_group = group

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for lt in self.model.layer_types:
            if lt == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
