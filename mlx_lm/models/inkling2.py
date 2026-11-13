# Copyright © 2026 Apple Inc.

"""Inkling 2.

https://huggingface.co/thinkingmachines/Inkling-2-Small
"""

import math
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import shard_linear, sum_gradients
from mlx.utils import tree_unflatten

from ..kernels import depth_attention, expert_routing, relative_attention
from .base import BaseModelArgs, gather_last_axis
from .cache import (
    ArraysCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from .muse_spark import shard_experts_inplace
from .pipeline import PipelineMixin
from .switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

_ROUTER_SCORE_EPS = 1e-20

_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0


_RENAMES = [
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.embed\.", "model.language_model.embed_tokens."),
    (r"^model\.llm\.", "model.language_model."),
    (r"\.attn_res\.mlp_queries\.(\d+)$", r".layers.\1.mlp_input.query"),
    (r"\.attn_res\.output_query$", ".final_input.query"),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.gate_attn\.", ".self_attn.gate_proj."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"sconv\.weight$", "sconv.conv1d.weight"),
    (r"convolution\.weight$", "convolution.conv1d.weight"),
    (r"sconv\.static_w$", "sconv.static_weight"),
    (r"sconv\.weight_proj$", "sconv.weight_proj.weight"),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"^model\.visual\.backbone\.ln_pre\.", "model.vision_tower.pre_layernorm."),
    (r"^model\.visual\.backbone\.", "model.vision_tower."),
    (r"^model\.visual\.proj_in\.", "model.vision_tower.merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "model.vision_tower.merger.linear_2."),
    (r"embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"(vision_tower\.layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(vision_tower\.layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$", r"\1.lambda_\2"),
]


def _rename(key: str) -> str:
    if m := re.match(r"model\.llm\.attn_res\.attn_queries\.(\d+)$", key):
        # Layer 0 has no attention input, so query i is for layer i + 1.
        layer = int(m.group(1)) + 1
        return f"model.language_model.layers.{layer}.attention_input.query"
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: list[int]
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    final_logit_softcapping: float


@dataclass
class VisionArgs(BaseModelArgs):
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    native_patch_size: int = 20
    adapter_merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = int(self.hidden_size * self.mlp_ratio)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    image_token_id: int = 200054

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        self.vision = None
        if self.vision_config:
            self.vision = VisionArgs.from_dict(self.vision_config)


def _use_kernels() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


def _precise_gelu(x: mx.array) -> mx.array:
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _l2_clip(hidden_states: mx.array, scale: mx.array, threshold: float) -> mx.array:
    states = hidden_states.astype(mx.float32) * scale
    norm = mx.sqrt((states * states).sum(axis=-1, keepdims=True))
    scaled = states * mx.minimum(threshold / (norm + 1e-6), 1.0)
    return scaled.astype(hidden_states.dtype)


@partial(mx.compile, shapeless=True)
def _softcapped_swiglu(
    gate: mx.array, up: mx.array, gate_softcap: float, up_softcap: float
) -> mx.array:
    input_dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate_capped = mx.tanh(gate / gate_softcap) * gate_softcap
    up_capped = mx.tanh(up / up_softcap) * up_softcap
    return (gate_capped * mx.sigmoid(gate) * up_capped).astype(input_dtype)


class L2Clip(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, 1.0, self.threshold)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, head_dim))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, self.scale, self.threshold)


class RelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states: mx.array) -> mx.array:
        return relative_states @ self.proj.astype(relative_states.dtype)


def _conv_input(
    hidden_states: mx.array,
    conv_state: mx.array | None,
    conv_mask: mx.array | None,
    kernel_size: int,
) -> mx.array:
    hidden_states = hidden_states.astype(mx.float32)
    if conv_mask is not None:
        hidden_states = mx.where(conv_mask[..., None], hidden_states, 0)
    if conv_state is None:
        shape = (hidden_states.shape[0], kernel_size - 1, hidden_states.shape[-1])
        conv_state = mx.zeros(shape, mx.float32)
    return mx.concatenate([conv_state, hidden_states], axis=1)


def _conv_state(window: mx.array, lengths: mx.array | None, n: int) -> mx.array:
    if lengths is None:
        pos = mx.arange(window.shape[1] - n, window.shape[1])[None, :, None]
    else:
        ends = mx.clip(lengths, 0, window.shape[1] - n)
        pos = (ends[:, None] + mx.arange(n))[..., None]
    # A gather copies, so the state does not keep the input in memory.
    return mx.take_along_axis(window, pos, axis=1)


class ShortConvolution(nn.Module):
    def __init__(self, dims: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dims, dims, kernel_size, groups=dims, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        K = self.kernel_size
        residual = hidden_states.astype(mx.float32)
        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        # As the reference, the conv runs in the weight dtype.
        out = self.conv1d(window.astype(self.conv1d.weight.dtype))
        out = (out.astype(mx.float32) + residual) / math.sqrt(K + 1)
        return out.astype(hidden_states.dtype), window


class DynamicShortConvolution(nn.Module):
    def __init__(self, dims: int, input_dims: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, dims))
        self.weight_proj = nn.Linear(input_dims, num_heads * kernel_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        dynamic_input: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        B, L, C = hidden_states.shape
        K, H = self.kernel_size, self.num_heads
        dynamic_weights = self.weight_proj(dynamic_input).astype(mx.float32)
        dynamic_weights = dynamic_weights.reshape(B, L, H, 1, K)
        static_weights = self.static_weight.astype(mx.float32).T.reshape(H, -1, K)
        weights = static_weights + dynamic_weights

        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        windows = mx.stack(
            [window[:, K - 1 - lag : K - 1 - lag + L] for lag in range(K)], axis=-1
        )
        windows = windows.reshape(B, L, H, -1, K)
        out = (windows * weights).sum(axis=-1).reshape(B, L, C)
        out = (out + hidden_states.astype(mx.float32)) / math.sqrt(2 * K + 1)
        return out.astype(hidden_states.dtype), window


def _positions(offset, length: int) -> mx.array:
    pos = mx.arange(length)
    if isinstance(offset, mx.array):
        return offset[:, None] + pos
    return pos + offset


def _key_positions(cache: Any, num_keys: int, num_queries: int, end) -> mx.array:
    k = mx.arange(num_keys)
    if num_queries == 1 and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):
        # A decode step writes in place, so the keys can be rotated.
        k = (k - cache._idx) % num_keys
    if isinstance(end, mx.array):
        return (end - num_keys)[:, None] + k
    return k + (end - num_keys)


def _attention_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    position_bias: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    query_positions: mx.array,
    key_positions: mx.array,
) -> mx.array:
    B, Hq, L, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    R = position_bias.shape[-1]
    distance = query_positions[..., :, None] - key_positions[..., None, :]
    visible = (distance >= 0) & (key_positions[..., None, :] >= 0)
    if window:
        visible = visible & (distance < window)
    if distance.ndim == 3:
        distance, visible = distance[:, None], visible[:, None]
    bias = mx.take_along_axis(
        position_bias.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    bias = mx.where(distance < R, bias.astype(mx.float32), 0.0)
    attention_mask = mx.where(visible, bias, -mx.inf).reshape(B, Hkv, -1, L, S)

    query = query.astype(mx.float32).reshape(B, Hkv, -1, L, D)
    key = key.astype(mx.float32)[:, :, None]
    attn_weights = (query @ key.swapaxes(-1, -2)) * scale + attention_mask
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    attn_weights = mx.softmax(combined_logits, axis=-1)[..., :S]
    attn_output = attn_weights.astype(value.dtype) @ value[:, :, None]
    return attn_output.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, *, sliding: bool):
        super().__init__()
        if sliding:
            self.num_heads = args.swa_num_attention_heads
            self.num_key_value_heads = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.sliding_window = args.sliding_window_size
            rel_extent = args.sliding_window_size
        else:
            self.num_heads = args.num_attention_heads
            self.num_key_value_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.sliding_window = 0
            rel_extent = args.rel_extent
        self.scaling = self.head_dim**-0.5

        dim = args.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.r_proj = nn.Linear(dim, self.num_heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, q_dim, bias=False)
        self.sink_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.rel_logits_proj = RelativeLogits(args.d_rel, rel_extent)
        self.q_l2_clip = ScaledL2Clip(
            self.num_heads, self.head_dim, args.qk_l2_clip_threshold
        )
        self.k_l2_clip = L2Clip(args.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(
            self.num_heads, args.d_rel, args.rel_l2_clip_threshold
        )
        self.eps = args.rms_norm_eps

    def _attend(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        cache: Any | None,
        tau: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        query_states = query_states.reshape(B, L, self.num_heads, -1)
        key_states = key_states.reshape(B, L, self.num_key_value_heads, -1)
        value_states = value_states.reshape(B, L, self.num_key_value_heads, -1)
        relative_states = relative_states.reshape(B, L, self.num_heads, -1)
        query_states = self.q_l2_clip(query_states)
        key_states = self.k_l2_clip(key_states)
        value_states = _unweighted_rms_norm(
            value_states.astype(mx.float32), self.eps
        ).astype(value_states.dtype)
        relative_states = self.rel_l2_clip(relative_states)
        if tau is not None:
            query_states = (query_states.astype(mx.float32) * tau).astype(
                query_states.dtype
            )
            relative_states = (relative_states.astype(mx.float32) * tau).astype(
                relative_states.dtype
            )
        position_bias = self.rel_logits_proj(relative_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)
        if cache is not None:
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        attn_output = self._attention(
            query_states, key_states, value_states, position_bias, sinks, cache
        )

        attn_output = attn_output.reshape(B, L, -1).astype(mx.float32)
        attn_output = attn_output * mx.sigmoid(gate.astype(mx.float32))
        return self.o_proj(attn_output.astype(gate.dtype))

    def _attention(self, query, key, value, position_bias, sinks, cache) -> mx.array:
        B, _, L, _ = query.shape
        S = key.shape[2]
        end = cache.offset if cache is not None else L
        bf16 = query.dtype == key.dtype == value.dtype == mx.bfloat16
        if L > 1 and _use_kernels() and bf16:
            key_start = mx.broadcast_to(S - end, (B,))
            return relative_attention(
                query,
                key,
                value,
                position_bias,
                sinks,
                scale=self.scaling,
                window=self.sliding_window,
                key_start=key_start,
            )

        query_positions = _positions(end - L, L)
        key_positions = _key_positions(cache, S, L, end)
        if L == 1 and B == 1:
            # The fused attention takes the sinks of one row.
            distance = query_positions.reshape(1) - key_positions.reshape(S)
            R = position_bias.shape[-1]
            bias = mx.take(position_bias[0, 0], mx.clip(distance, 0, R - 1), axis=-1)
            bias = mx.where(distance < R, bias, mx.zeros_like(bias))
            visible = (distance >= 0) & (key_positions.reshape(S) >= 0)
            if self.sliding_window:
                visible = visible & (distance < self.sliding_window)
            mask = mx.where(visible, bias, -mx.inf).astype(query.dtype)
            attn_output = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scaling,
                mask=mask[None, :, None, :],
                sinks=sinks[0, 0].astype(query.dtype),
            )
            return attn_output.transpose(0, 2, 1, 3)

        return _attention_reference(
            query,
            key,
            value,
            position_bias,
            sinks,
            scale=self.scaling,
            window=self.sliding_window,
            query_positions=query_positions,
            key_positions=key_positions,
        )


class SlidingAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=True)
        dims = self.num_key_value_heads * self.head_dim
        self.k_sconv = ShortConvolution(dims, args.sconv_kernel_size)
        self.v_sconv = ShortConvolution(dims, args.sconv_kernel_size)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        key_states, conv_states[0] = self.k_sconv(
            self.k_proj(hidden_states), conv_states[0], conv_mask
        )
        value_states, conv_states[1] = self.v_sconv(
            self.v_proj(hidden_states), conv_states[1], conv_mask
        )
        return self._attend(
            hidden_states,
            self.q_proj(hidden_states),
            key_states,
            value_states,
            self.r_proj(hidden_states),
            kv_cache,
        )


class FullAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=False)
        dims = self.num_key_value_heads * self.head_dim
        conv_args = (args.hidden_size, self.num_key_value_heads, args.sconv_kernel_size)
        self.k_sconv = DynamicShortConvolution(dims, *conv_args)
        self.v_sconv = DynamicShortConvolution(dims, *conv_args)
        self.log_scaling_n_floor = args.log_scaling_n_floor
        self.log_scaling_alpha = args.log_scaling_alpha

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        key_states, conv_states[0] = self.k_sconv(
            self.k_proj(key_hidden_states),
            key_hidden_states,
            conv_states[0],
            conv_mask,
        )
        value_states, conv_states[1] = self.v_sconv(
            self.v_proj(value_hidden_states),
            value_hidden_states,
            conv_states[1],
            conv_mask,
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        position_ids = _positions(offset, hidden_states.shape[1]).astype(mx.float32)
        tau = 1.0 + self.log_scaling_alpha * mx.log(
            mx.maximum((position_ids + 1) / self.log_scaling_n_floor, 1.0)
        )
        return self._attend(
            hidden_states,
            self.q_proj(hidden_states),
            key_states,
            value_states,
            self.r_proj(hidden_states),
            kv_cache,
            tau=tau[..., None, None],
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_softcap = args.mlp_gate_softcap
        self.up_softcap = args.mlp_up_softcap

    def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
        return _softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap)


class Experts(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        dims, hidden = args.hidden_size, args.intermediate_size
        num_experts = args.n_routed_experts
        # The rows alternate between gate and up, as in the checkpoint.
        self.gate_up_proj = SwitchLinear(dims, 2 * hidden, num_experts, bias=False)
        self.down_proj = SwitchLinear(hidden, dims, num_experts, bias=False)
        self.act_fn = SoftcappedSwiGLU(args)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x = self.act_fn(gate_up[..., 0::2], gate_up[..., 1::2])
        x = self.down_proj(x, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class TopKRouter(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.route_scale
        num_experts, dims = args.n_routed_experts, args.hidden_size
        self.weight = mx.zeros((num_experts, dims), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        logits = hidden_states.astype(mx.float32) @ self.weight.T
        if _use_kernels():
            return expert_routing(
                logits,
                self.e_score_correction_bias,
                top_k=self.top_k,
                multiplier=1.0,
                score_eps=_ROUTER_SCORE_EPS,
                route_scale=self.routed_scaling_factor,
            )
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias
        k = self.top_k
        indices = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_SCORE_EPS)
        return indices, weights * self.routed_scaling_factor


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = TopKRouter(args)
        self.experts = Experts(args)
        self.sharding_group = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            hidden_states = sum_gradients(group)(hidden_states)
        selected_experts, routing_weights = self.gate(hidden_states)
        expert_states = self.experts(hidden_states, selected_experts)
        expert_states = expert_states.astype(mx.float32) * routing_weights[..., None]
        final_hidden_states = expert_states.sum(axis=-2)
        if group is not None:
            final_hidden_states = mx.distributed.all_sum(
                final_hidden_states, group=group
            )
        return final_hidden_states.astype(hidden_states.dtype)


class BlockAttentionResidual(nn.Module):
    def __init__(self, dims: int, num_queries: int = 1):
        super().__init__()
        shape = (dims,) if num_queries == 1 else (num_queries, dims)
        self.query = mx.zeros(shape, dtype=mx.float32)

    def __call__(
        self, block_states: mx.array, partial_block_states: mx.array
    ) -> mx.array:
        dims = partial_block_states.shape[-1]
        query = self.query.reshape(-1, dims).astype(mx.float32)
        scale = dims**-0.5
        if _use_kernels():
            return depth_attention(
                block_states, partial_block_states, query, scale=scale
            )
        stacked_states = mx.concatenate([block_states, partial_block_states[None]])
        stacked_states = mx.moveaxis(stacked_states, 0, -2)
        weights = mx.softmax((stacked_states @ query.T) * scale, axis=-2, precise=True)
        return weights.swapaxes(-1, -2) @ stacked_states


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int, *, num_queries: int):
        super().__init__()
        dims = args.hidden_size
        self.is_block_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        self.attention_input = (
            BlockAttentionResidual(dims, num_queries) if layer_idx > 0 else None
        )
        self.mlp_input = BlockAttentionResidual(dims)
        self.mlp = SparseMoeBlock(args)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        kernel_size = args.sconv_kernel_size
        self.post_attention_convolution = ShortConvolution(dims, kernel_size)
        self.post_mlp_convolution = ShortConvolution(dims, kernel_size)

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: mx.array | None,
        *,
        cache: Any | None = None,
        conv_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        dtype = hidden_states.dtype
        conv = cache[1] if cache is not None else None
        conv_states = list(conv.cache) if conv is not None else [None] * 4
        block_residual = hidden_states
        normed_residual = _unweighted_rms_norm(block_residual.astype(mx.float32))

        attention_inputs = None
        if self.attention_input is not None:
            attention_inputs = self.attention_input(block_states, normed_residual)
        hidden_states = self._self_attention(
            hidden_states, attention_inputs, cache, conv_states, conv_mask
        )
        hidden_states, conv_states[2] = self.post_attention_convolution(
            hidden_states, conv_states[2], conv_mask
        )

        residual = hidden_states = hidden_states.astype(mx.float32)
        if self.is_block_boundary:
            block = normed_residual[None]
            if block_states is None:
                block_states = block
            else:
                block_states = mx.concatenate([block_states, block])
        else:
            residual = hidden_states = block_residual.astype(mx.float32) + hidden_states

        normed_states = _unweighted_rms_norm(hidden_states)
        hidden_states = self.mlp_input(block_states, normed_states)[..., 0, :]
        hidden_states = self.mlp(
            self.post_attention_layernorm(hidden_states).astype(dtype)
        )
        hidden_states, conv_states[3] = self.post_mlp_convolution(
            hidden_states, conv_states[3], conv_mask
        )

        if conv is not None:
            n = self.post_mlp_convolution.kernel_size - 1
            for i, window in enumerate(conv_states):
                conv[i] = _conv_state(window, conv.lengths, n)
            conv.advance(block_residual.shape[1])
        hidden_states = residual + hidden_states.astype(mx.float32)
        return hidden_states.astype(dtype), block_states


class SlidingDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=1)
        self.self_attn = SlidingAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _self_attention(
        self, hidden_states, attention_inputs, cache, conv_states, conv_mask
    ):
        dtype = hidden_states.dtype
        if attention_inputs is not None:
            hidden_states = attention_inputs[..., 0, :]
        return self.self_attn(
            self.input_layernorm(hidden_states).astype(dtype),
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )


class FullDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=3)
        self.self_attn = FullAttention(args)
        eps = args.rms_norm_eps
        self.q_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.k_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.v_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def _self_attention(
        self, hidden_states, attention_inputs, cache, conv_states, conv_mask
    ):
        dtype = hidden_states.dtype
        if attention_inputs is None:
            query_states = key_states = value_states = hidden_states
        else:
            query_states, key_states, value_states = (
                attention_inputs[..., i, :] for i in range(3)
            )
        return self.self_attn(
            self.q_input_layernorm(query_states).astype(dtype),
            self.k_input_layernorm(key_states).astype(dtype),
            self.v_input_layernorm(value_states).astype(dtype),
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # Not in embed_tokens as in transformers, so that it can be quantized.
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            (
                SlidingDecoderLayer(args, i)
                if i in args.local_layer_ids
                else FullDecoderLayer(args, i)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.final_input = BlockAttentionResidual(args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is None:
            input_embeddings = self.embed(inputs)
        hidden_states = input_embeddings
        dtype = hidden_states.dtype

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)
        conv_mask = None
        if cache[0] is not None:
            conv_mask = cache[0][1].make_mask(hidden_states.shape[1])

        rank, size = self.pipeline_rank, self.pipeline_size
        block_states = None
        if rank < size - 1:
            num_blocks = -(-self.start_idx // self.args.n_layers_per_attn_res_block)
            states = mx.zeros((num_blocks + 1, *hidden_states.shape), mx.float32)
            states = mx.distributed.recv_like(states, rank + 1)
            hidden_states, block_states = states[0].astype(dtype), states[1:]

        for layer, c in zip(layers, cache):
            hidden_states, block_states = layer(
                hidden_states, block_states, cache=c, conv_mask=conv_mask
            )

        if rank != 0:
            states = mx.concatenate(
                [hidden_states.astype(mx.float32)[None], block_states]
            )
            states = mx.distributed.send(states, (rank - 1) % size)
            if cache[-1] is not None:
                kv_cache = cache[-1][0]
                kv_cache.keys = mx.depends(kv_cache.keys, states)
            # Only the output of rank 0 is used.
            hidden_states = states[0].astype(dtype)
        else:
            normed_states = _unweighted_rms_norm(hidden_states.astype(mx.float32))
            hidden_states = self.final_input(block_states, normed_states)[..., 0, :]
            hidden_states = self.norm(hidden_states).astype(dtype)

        if size > 1:
            batch = hidden_states.shape[0]
            hidden_states = mx.distributed.all_gather(hidden_states)[:batch]
        return hidden_states


def _grid_positions(grid_h: int, grid_w: int, merge: int) -> tuple[mx.array, mx.array]:
    i = mx.arange(grid_h * grid_w)
    blocks_w = grid_w // merge
    row = i // (merge * merge * blocks_w) * merge + i // merge % merge
    col = i // (merge * merge) % blocks_w * merge + i % merge
    return row, col


def _bilinear_taps(index: mx.array, size: int, side: int) -> tuple[mx.array, mx.array]:
    src = (index.astype(mx.float32) + 0.5) * side / size - 0.5
    taps = mx.floor(src)[:, None] + mx.arange(2)
    weights = mx.maximum(1.0 - mx.abs(src[:, None] - taps), 0.0)
    return mx.clip(taps, 0, side - 1).astype(mx.int32), weights


class VisionEmbeddings(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.merge_size = args.adapter_merge_size
        self.grid_size = args.posemb_grid_size
        patch_dims = args.num_channels * args.native_patch_size**2
        self.patch_embedding = nn.Linear(patch_dims, args.hidden_size, bias=False)
        self.position_embedding = nn.Embedding(
            args.posemb_grid_size**2, args.hidden_size
        )
        self.class_embedding = mx.zeros((args.hidden_size,))
        self.cls_position_embedding = mx.zeros((args.hidden_size,))

    def __call__(self, pixel_values: mx.array, grid_h: int, grid_w: int) -> mx.array:
        patch_embeddings = self.patch_embedding(pixel_values)

        side = self.grid_size
        row, col = _grid_positions(grid_h, grid_w, self.merge_size)
        row_taps, row_weights = _bilinear_taps(row, grid_h, side)
        col_taps, col_weights = _bilinear_taps(col, grid_w, side)
        interp_indices = row_taps[:, :, None] * side + col_taps[:, None, :]
        interp_weights = row_weights[:, :, None] * col_weights[:, None, :]
        position_embeddings = self.position_embedding(interp_indices.reshape(-1, 4))
        position_embeddings = position_embeddings.astype(mx.float32)
        position_embeddings = position_embeddings * interp_weights.reshape(-1, 4, 1)
        patch_embeddings = patch_embeddings + position_embeddings.sum(axis=1).astype(
            patch_embeddings.dtype
        )

        cls_embedding = self.class_embedding + self.cls_position_embedding
        cls_embedding = cls_embedding.astype(patch_embeddings.dtype)
        return mx.concatenate([cls_embedding[None], patch_embeddings])


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims = args.head_dim // 2
        self._inv_freq = 1.0 / (
            args.rope_theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )

    def __call__(self, row: mx.array, col: mx.array) -> tuple[mx.array, mx.array]:
        freqs_h = row[:, None].astype(mx.float32) * self._inv_freq
        freqs_w = col[:, None].astype(mx.float32) * self._inv_freq
        freqs = mx.repeat(mx.concatenate([freqs_w, freqs_h], axis=-1), 2, axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


def _apply_rotary_pos_emb_vision(
    hidden_states: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    states = hidden_states.astype(mx.float32)
    pairs = states.reshape(*states.shape[:-1], -1, 2)
    rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(states.shape)
    return (states * cos + rotated * sin).astype(hidden_states.dtype)


class VisionAttention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.scaling = args.head_dim**-0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(
        self, hidden_states: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        N = hidden_states.shape[0]
        shape = (N, self.num_heads, -1)
        cos, sin = cos[:, None], sin[:, None]
        query_states = self.q_proj(hidden_states).reshape(shape)
        key_states = self.k_proj(hidden_states).reshape(shape)
        value_states = self.v_proj(hidden_states).reshape(shape)
        query_states = _apply_rotary_pos_emb_vision(query_states, cos, sin)
        key_states = _apply_rotary_pos_emb_vision(key_states, cos, sin)
        query_states, key_states, value_states = (
            states.transpose(1, 0, 2)[None]
            for states in (query_states, key_states, value_states)
        )
        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )
        attn_output = attn_output[0].transpose(1, 0, 2).reshape(N, -1)
        return self.out_proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(_precise_gelu(self.fc1(hidden_states)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims, eps = args.hidden_size, args.layer_norm_eps
        self.self_attn = VisionAttention(args)
        self.mlp = VisionMLP(args)
        self.layernorm_before = nn.LayerNorm(dims, eps=eps)
        self.layernorm_after = nn.LayerNorm(dims, eps=eps)
        self.lambda_1 = mx.ones((dims,))
        self.lambda_2 = mx.ones((dims,))

    def __call__(
        self, hidden_states: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.layernorm_before(hidden_states), cos, sin)
        hidden_states = residual + self.lambda_1 * hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layernorm_after(hidden_states))
        return residual + self.lambda_2 * hidden_states


class VisionMerger(nn.Module):
    def __init__(self, args: VisionArgs, text_hidden_size: int):
        super().__init__()
        dims = args.adapter_merge_size**2 * args.hidden_size
        hidden, bias = args.projection_hidden_dim, args.projector_bias
        self.linear_1 = nn.Linear(dims, hidden, bias=bias)
        self.linear_2 = nn.Linear(hidden, text_hidden_size, bias=bias)

    def __call__(self, image_features: mx.array) -> mx.array:
        return self.linear_2(_precise_gelu(self.linear_1(image_features)))


class VisionModel(nn.Module):
    def __init__(self, args: VisionArgs, text_hidden_size: int):
        super().__init__()
        self.args = args
        self.embeddings = VisionEmbeddings(args)
        self.pre_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.rotary_emb = VisionRotaryEmbedding(args)
        self.layers = [VisionEncoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.merger = VisionMerger(args, text_hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        C, H, W = pixel_values.shape
        p, m = self.args.native_patch_size, self.args.adapter_merge_size
        if H % (p * m) or W % (p * m):
            raise ValueError(f"Image size {H}x{W} is not a multiple of {p * m}.")
        rows, cols = H // (p * m), W // (p * m)
        patches = pixel_values.transpose(1, 2, 0).reshape(rows, m, p, cols, m, p, C)
        # The merge blocks in raster order, then the patches of each block.
        patches = patches.transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, p * p * C)
        dtype = self.embeddings.patch_embedding.weight.dtype
        hidden_states = self.embeddings(patches.astype(dtype), rows * m, cols * m)
        hidden_states = self.pre_layernorm(hidden_states)

        row, col = _grid_positions(rows * m, cols * m, m)
        zero = mx.zeros((1,), dtype=row.dtype)
        cos, sin = self.rotary_emb(
            mx.concatenate([zero, row + 1]), mx.concatenate([zero, col + 1])
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        patch_states = hidden_states[1:].reshape(rows * cols, -1)
        return self.merger(patch_states)


def _splice_image_features(
    inputs_embeds: mx.array,
    input_ids: mx.array,
    image_features: mx.array,
    image_token_id: int,
) -> mx.array:
    if image_features.shape[0] == 0:
        return inputs_embeds
    special_image_mask = input_ids == image_token_id
    idx = mx.cumsum(special_image_mask.flatten()).reshape(special_image_mask.shape) - 1
    idx = mx.clip(idx, 0, image_features.shape[0] - 1)
    rows = image_features[idx].astype(inputs_embeds.dtype)
    return mx.where(special_image_mask[..., None], rows, inputs_embeds)


class Inkling2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = TextModel(args.text)
        if args.vision is not None:
            self.vision_tower = VisionModel(args.vision, args.text.hidden_size)

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self) -> list[DecoderLayer | None]:
        return self.language_model.layers

    @property
    def start_idx(self) -> int:
        return self.language_model.start_idx

    @property
    def end_idx(self) -> int | None:
        return self.language_model.end_idx

    def pipeline(self, group, split=None) -> None:
        self.language_model.pipeline(group, split)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.sharding_group = None

    def encode_image(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        """pixel_values: one image (num_channels, H, W) or a list of images of
        any size. Returns the rows of the image tokens of all images."""
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        assert len(pixel_values), "No images."
        return mx.concatenate([self.model.vision_tower(p) for p in pixel_values])

    def embed_inputs(
        self,
        inputs: mx.array,
        *,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        inputs_embeds = self.model.language_model.embed(inputs)
        if pixel_values is not None:
            image_features = self.encode_image(pixel_values)
            inputs_embeds = _splice_image_features(
                inputs_embeds, inputs, image_features, self.args.image_token_id
            )
        return inputs_embeds

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        if pixel_values is not None:
            assert input_embeddings is None, "Pass pixel_values or input_embeddings."
            input_embeddings = self.embed_inputs(inputs, pixel_values=pixel_values)
        hidden_states = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        logits = self.lm_head(hidden_states)
        if self.sharding_group is not None:
            logits = gather_last_axis(logits, self.sharding_group)
        text = self.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        experts = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
        amax, denominator = {}, {}
        sanitized = {}
        for k, v in weights.items():
            if k.startswith(("model.llm.", "model.visual.")):
                k = _rename(k)
                if k.endswith("conv1d.weight"):
                    v = v.swapaxes(1, 2)
            if k.startswith("model.vision_tower.") and self.args.vision is None:
                continue
            if k.endswith(".mlp.gate.weight"):
                v = v.astype(mx.float32)
            elif m := experts.match(k):
                prefix, which, suffix = m.groups()
                name = f"{prefix}.{'gate_up_proj' if which == 'w13' else 'down_proj'}"
                if suffix == "_amax":
                    amax[name] = v
                elif suffix == "_outer_scale_denominator":
                    denominator[name] = v
                elif suffix == "_scale":
                    sanitized[f"{name}.scales"] = v
                else:
                    sanitized[f"{name}.weight"] = v.view(mx.uint32)
                continue
            sanitized[k] = v

        # Pipeline stages can miss the files of the other layers.
        for name in amax.keys() & denominator.keys():
            scale = _NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX
            scale = scale / denominator[name].astype(mx.float32)
            sanitized[f"{name}.global_scale"] = amax[name].astype(mx.float32) * scale

        # mlx-lm does not read this quantization config, so set the experts here.
        modes = {(4, 16): "nvfp4", (4, 32): "mxfp4", (8, 32): "mxfp8"}
        quantized = []
        for path, module in self.named_modules():
            scales = sanitized.get(f"{path}.scales")
            if not isinstance(module, SwitchLinear) or scales is None:
                continue
            if scales.dtype != mx.uint8:
                continue
            in_dims = module.weight.shape[-1]
            group_size = in_dims // scales.shape[-1]
            bits = sanitized[f"{path}.weight"].shape[-1] * 32 // in_dims
            if (bits, group_size) not in modes:
                continue
            ql = module.to_quantized(
                group_size,
                bits,
                mode=modes[bits, group_size],
                global_scale=f"{path}.global_scale" in sanitized,
            )
            quantized.append((path, ql))
        if quantized:
            self.update_modules(tree_unflatten(quantized))
        return sanitized

    def shard(self, group: mx.distributed.Group | None = None) -> None:
        group = group or mx.distributed.init()
        N, r = group.size(), group.rank()
        text = self.args.text
        for n_kv_heads in (text.num_key_value_heads, text.swa_num_key_value_heads):
            assert (
                n_kv_heads % N == 0
            ), f"group size {N} must divide num_key_value_heads {n_kv_heads}"
        assert (
            text.vocab_size % N == 0
        ), f"group size {N} must divide vocab_size {text.vocab_size}"

        def part(w: mx.array, axis: int) -> mx.array:
            return mx.contiguous(mx.split(w, N, axis=axis)[r])

        self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
        self.sharding_group = group
        for layer in self.layers:
            attn = layer.self_attn
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "r_proj",
                "gate_proj",
                "sink_proj",
            ):
                proj = shard_linear(getattr(attn, name), "all-to-sharded", group=group)
                setattr(attn, name, proj)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.q_l2_clip.scale = part(attn.q_l2_clip.scale, 0)
            attn.rel_l2_clip.scale = part(attn.rel_l2_clip.scale, 0)
            for conv in (attn.k_sconv, attn.v_sconv):
                if isinstance(conv, ShortConvolution):
                    conv.conv1d.weight = part(conv.conv1d.weight, 0)
                    conv.conv1d.groups //= N
                else:
                    conv.static_weight = part(conv.static_weight, 1)
                    conv.weight_proj = shard_linear(
                        conv.weight_proj, "all-to-sharded", group=group
                    )
                    conv.num_heads //= N
            attn.num_heads //= N
            attn.num_key_value_heads //= N

            layer.mlp.sharding_group = group
            experts = layer.mlp.experts
            shard_experts_inplace(experts.gate_up_proj, "all-to-sharded", group)
            shard_experts_inplace(experts.down_proj, "sharded-to-all", group)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:

        def predicate(path: str, module: nn.Module) -> bool:
            return not path.startswith("model.vision_tower.")

        return predicate

    @property
    def cast_predicate(self) -> Callable[[str], bool]:

        def predicate(k: str) -> bool:
            return not k.endswith(
                (".query", ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
            )

        return predicate

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.language_model.pipeline_layers

    def make_cache(self) -> list[CacheList]:
        window = self.args.text.sliding_window_size
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=window)
                    if isinstance(layer, SlidingDecoderLayer)
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for layer in self.layers
        ]
