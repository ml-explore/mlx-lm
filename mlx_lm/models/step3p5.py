# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwiGLU, SwitchGLU


@partial(mx.compile, shapeless=True)
def clamped_swiglu(x, gate, limit):
    gate = mx.clip(nn.silu(gate), a_min=None, a_max=limit)
    x = mx.clip(x, a_min=-limit, a_max=limit)
    return gate * x


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return clamped_swiglu(x, gate, self.limit)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    max_position_embeddings: int = 262144
    sliding_window: int = 512
    layer_types: Optional[List[str]] = None
    yarn_only_types: Optional[List[str]] = None
    partial_rotary_factors: Optional[List[float]] = None
    attention_other_setting: Optional[Dict] = None
    use_head_wise_attn_gate: bool = True
    moe_num_experts: int = 288
    moe_top_k: int = 8
    moe_intermediate_size: int = 1280
    share_expert_dim: int = 1280
    moe_layers_enum: Optional[str] = None
    moe_router_scaling_factor: float = 3.0
    norm_expert_weight: bool = True
    swiglu_limits: Optional[List[float]] = None
    swiglu_limits_shared: Optional[List[float]] = None
    tie_word_embeddings: bool = False


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Step3p5MLP(nn.Module):
    def __init__(
        self, args: ModelArgs, intermediate_size: int, swiglu_limit: float = 0
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.limit = swiglu_limit if swiglu_limit and swiglu_limit > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        if self.limit is not None:
            return self.down_proj(
                clamped_swiglu(self.up_proj(x), self.gate_proj(x), self.limit)
            )
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def moe_gate_select(gates, router_bias, top_k, routed_scaling_factor, norm_topk_prob):
    scores = mx.sigmoid(gates.astype(mx.float32))
    corrected_scores = scores + router_bias

    topk_indices = mx.argpartition(-corrected_scores, kth=top_k - 1, axis=-1)[
        ..., :top_k
    ]
    topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

    if norm_topk_prob:
        topk_weights = topk_weights / (
            mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
        )

    return topk_indices, topk_weights * routed_scaling_factor


class Step3p5MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.moe_top_k
        self.n_routed_experts = args.moe_num_experts
        self.routed_scaling_factor = args.moe_router_scaling_factor
        self.norm_topk_prob = args.norm_expert_weight

        self.gate = nn.Linear(args.hidden_size, self.n_routed_experts, bias=False)
        self.router_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x: mx.array):
        return moe_gate_select(
            self.gate(x),
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Step3p5MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        swiglu_limit = 0
        if args.swiglu_limits and layer_idx < len(args.swiglu_limits):
            swiglu_limit = args.swiglu_limits[layer_idx] or 0

        swiglu_limit_shared = 0
        if args.swiglu_limits_shared and layer_idx < len(args.swiglu_limits_shared):
            swiglu_limit_shared = args.swiglu_limits_shared[layer_idx] or 0

        self.gate = Step3p5MoEGate(args)

        activation = ClampedSwiGLU(swiglu_limit) if swiglu_limit > 0 else SwiGLU()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.moe_num_experts,
            activation=activation,
        )

        self.share_expert = Step3p5MLP(
            args,
            intermediate_size=args.share_expert_dim,
            swiglu_limit=swiglu_limit_shared,
        )

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        topk_indices, topk_weights = self.gate(x)
        routed_output = self.switch_mlp(x, topk_indices)
        routed_output = (
            (routed_output * topk_weights[..., None])
            .sum(axis=-2)
            .astype(routed_output.dtype)
        )
        y = routed_output + self.share_expert(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class Step3p5Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size

        layer_types = args.layer_types or []
        if layer_types:
            self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        else:
            self.is_sliding = layer_idx % 2 == 0

        if self.is_sliding and args.attention_other_setting:
            self.num_heads = args.attention_other_setting["num_attention_heads"]
            self.num_kv_heads = args.attention_other_setting["num_attention_groups"]
        else:
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_attention_groups

        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.use_head_wise_attn_gate = args.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(dim, self.num_heads, bias=False)

        rope_theta = args.rope_theta
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[layer_idx]

        partial_rotary_factor = 1.0
        if args.partial_rotary_factors and layer_idx < len(args.partial_rotary_factors):
            partial_rotary_factor = args.partial_rotary_factors[layer_idx]

        rope_dims = int(self.head_dim * partial_rotary_factor)

        yarn_only_types = args.yarn_only_types or []
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None
        else:
            rope_scaling = args.rope_scaling

        self.rope = initialize_rope(
            dims=rope_dims,
            base=rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.num_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.num_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

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
        output = output.transpose(0, 2, 1, 3)

        if self.use_head_wise_attn_gate:
            output = output * mx.sigmoid(self.g_proj(x))[..., None]

        return self.o_proj(output.reshape(B, L, -1))


class Step3p5DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.self_attn = Step3p5Attention(args, layer_idx)
        self.is_sliding = self.self_attn.is_sliding

        moe_layers_idx = set()
        if args.moe_layers_enum:
            moe_layers_idx = {int(i) for i in args.moe_layers_enum.strip().split(",")}
        else:
            moe_layers_idx = set(range(1, args.num_hidden_layers))

        self.is_moe_layer = layer_idx in moe_layers_idx

        if self.is_moe_layer:
            self.mlp = Step3p5MoE(args, layer_idx)
        else:
            swiglu_limit = 0
            if args.swiglu_limits_shared and layer_idx < len(args.swiglu_limits_shared):
                swiglu_limit = args.swiglu_limits_shared[layer_idx] or 0
            self.mlp = Step3p5MLP(
                args,
                intermediate_size=args.intermediate_size,
                swiglu_limit=swiglu_limit,
            )

        self.input_layernorm = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Step3p5Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Step3p5DecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._swa_idx = next(
            (i for i, l in enumerate(self.layers) if l.is_sliding), None
        )
        self._full_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_sliding), None
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * self.num_layers

        full_mask = None
        swa_mask = None

        if self._full_idx is not None:
            full_mask = create_attention_mask(h, cache[self._full_idx])

        if self._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding else full_mask
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Step3p5Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]

        is_vanilla = any(
            src in k and dst not in k for k in weights for src, dst in remappings
        )

        new_weights = {}
        for k, v in weights.items():
            if ".mtp" in k:
                continue
            if "model.layers." in k:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    if int(parts[2]) >= self.args.num_hidden_layers:
                        continue

            for src, dst in remappings:
                if src in k and dst not in k:
                    k = k.replace(src, dst)
                    break

            if is_vanilla and k.endswith(".weight") and "norm" in k:
                v = v + 1

            new_weights[k] = v

        return new_weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return "router_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "mlp.gate.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_heads //= N
            layer.self_attn.num_kv_heads //= N

            if layer.self_attn.use_head_wise_attn_gate:
                layer.self_attn.g_proj = shard_linear(
                    layer.self_attn.g_proj, "all-to-sharded", group=group
                )

            if isinstance(layer.mlp, Step3p5MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.share_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.share_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.share_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
