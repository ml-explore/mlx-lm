# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    num_experts_per_tok: int
    hybrid_layer_pattern: List[int]
    moe_layer_freq: List[int]
    add_swa_attention_sink_bias: bool
    add_full_attention_sink_bias: bool
    sliding_window_size: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: Optional[float]
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    max_position_embeddings: int
    layernorm_epsilon: float
    rope_theta: float
    swa_rope_theta: float
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    swa_head_dim: int
    swa_v_head_dim: int
    partial_rotary_factor: int


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sliding_window: bool):
        super().__init__()

        dim = args.hidden_size
        self.is_sliding_window = is_sliding_window
        if self.is_sliding_window:
            self.n_heads = n_heads = args.swa_num_attention_heads
            self.n_kv_heads = n_kv_heads = args.swa_num_key_value_heads
            self.has_sinks = args.add_swa_attention_sink_bias
            head_dim = args.swa_head_dim
            v_head_dim = args.swa_v_head_dim
            rope_theta = args.swa_rope_theta
        else:
            self.n_heads = n_heads = args.num_attention_heads
            self.n_kv_heads = n_kv_heads = args.num_key_value_heads
            self.has_sinks = args.add_full_attention_sink_bias
            head_dim = args.head_dim
            v_head_dim = args.v_head_dim
            rope_theta = args.rope_theta

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * v_head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * v_head_dim, dim, bias=False)
        if self.has_sinks:
            self.attention_sink_bias = mx.ones((self.n_heads,))
        else:
            self.attention_sink_bias = None

        self.rope = nn.RoPE(
            int(args.partial_rotary_factor * head_dim),
            traditional=False,
            base=rope_theta,
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
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attention_sink_bias,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(
        self, config: ModelArgs, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        down_proj = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = (
            config.routed_scaling_factor
            if config.routed_scaling_factor is not None
            else 1.0
        )
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(
                config=config, intermediate_size=intermediate_size
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_moe, is_sliding_window):
        super().__init__()
        self.self_attn = Attention(config, is_sliding_window)
        self.mlp = MoE(config) if is_moe else MLP(config)
        self.is_sliding_window = is_sliding_window
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(
                config,
                is_moe=config.moe_layer_freq[idx] == 1,
                is_sliding_window=config.hybrid_layer_pattern[idx] == 1,
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.swa_idx = config.hybrid_layer_pattern.index(1)
        self.ga_idx = config.hybrid_layer_pattern.index(0)
        self.sliding_window_size = config.sliding_window_size

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(x, cache[self.ga_idx])
        swa_mask = create_attention_mask(
            x, cache[self.swa_idx], window_size=self.sliding_window_size
        )

        for l, c in zip(self.layers, cache):
            mask = swa_mask if l.is_sliding_window else full_mask
            h = l(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = LanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            dtype = weight.dtype
            bs = 128  # block size
            m, n = weight.shape
            pad_bottom = bs * scale_inv.shape[0] - m
            pad_side = bs * scale_inv.shape[1] - n
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        # Dequantize fp8
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # Stack experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        # Remove multi-token prediction layer
        return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size))
            else:
                caches.append(KVCache())
        return caches
