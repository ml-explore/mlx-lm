# Copyright © 2025 Apple Inc.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache, MambaCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False
    use_cache: bool = True

    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = False

    linear_attn_config: Optional[Dict[str, Any]] = None

    num_experts: Optional[int] = None
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_intermediate_size: Optional[int] = None
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1

    model_max_length: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.linear_attn_config is not None:
            for key in ("head_dim", "num_heads", "short_conv_kernel_size"):
                if key not in self.linear_attn_config:
                    raise ValueError(
                        f"linear_attn_config missing required field '{key}'"
                    )

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 0

    @property
    def is_mla(self) -> bool:
        return (
            any(
                getattr(self, name) is not None
                for name in (
                    "q_lora_rank",
                    "kv_lora_rank",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                )
            )
            or self.mla_use_nope
        )

    @property
    def is_linear_attn(self) -> bool:
        return self.linear_attn_config is not None

    def is_kda_layer(self, layer_idx: int) -> bool:
        if not self.is_linear_attn:
            return False
        kda_layers = self.linear_attn_config.get("kda_layers")
        if not kda_layers:
            return False
        return (layer_idx + 1) in kda_layers


@partial(mx.compile, shapeless=True)
def _swish(x: mx.array) -> mx.array:
    return nn.silu(x)


def _activation_fn(name: str):
    if name == "silu" or name == "swish":
        return _swish
    raise ValueError(f"Unsupported activation function '{name}' for KimiLinear")


class RMSNormGated(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-5, activation: str = "sigmoid"
    ):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps
        if activation not in ("sigmoid", "silu", "swish"):
            raise ValueError(f"Unsupported gate activation '{activation}'")
        self.activation = activation

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        if self.activation == "sigmoid":
            gate = mx.sigmoid(gate)
        else:
            gate = nn.silu(gate)
        return y * gate


class KimiMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act = _activation_fn(args.hidden_act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    elif score_function == "softmax":
        scores = mx.softmax(gates.astype(mx.float32), axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class KimiSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        if experts is None:
            raise ValueError("num_experts must be specified for MoE layers")

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, args.moe_intermediate_size, experts)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        out = self.switch_mlp(x, inds)
        out = (out * weights[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class KimiMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(
            hidden,
            (args.kv_lora_rank or 0) + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_norm = nn.RMSNorm(
            args.kv_lora_rank or self.qk_nope_head_dim, eps=args.rms_norm_eps
        )
        self.kv_b_proj = nn.Linear(
            args.kv_lora_rank or self.qk_nope_head_dim,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

        rope_dim = self.qk_rope_head_dim or self.q_head_dim
        self.rope = initialize_rope(
            rope_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.model_max_length or 1_000_000,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q_states = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q_pass, q_rot = mx.split(q_states, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj(x)
        k_pass, k_rot = mx.split(
            compressed, [compressed.shape[-1] - self.qk_rope_head_dim], axis=-1
        )
        k_pass = self.kv_a_norm(k_pass)
        kv = self.kv_b_proj(k_pass)
        kv = kv.reshape(
            B,
            L,
            self.num_heads,
            self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim,
        )
        k_pass, v_states = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if self.qk_rope_head_dim:
            k_rot = mx.reshape(k_rot, (B, L, 1, self.qk_rope_head_dim))
            k_rot = mx.broadcast_to(k_rot, (*k_pass.shape[:-1], self.qk_rope_head_dim))
        else:
            k_rot = mx.zeros((*k_pass.shape[:-1], 0), dtype=k_pass.dtype)

        queries = mx.concatenate([q_pass, q_rot], axis=-1).transpose(0, 2, 1, 3)
        keys = mx.concatenate([k_pass, k_rot], axis=-1).transpose(0, 2, 1, 3)
        values = v_states.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        out = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache,
            scale=self.scale,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self, x: mx.array, cache: Optional[mx.array]
    ) -> Tuple[mx.array, mx.array]:
        if cache is None:
            pad = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        else:
            pad = cache
        conv_input = mx.concatenate([pad, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        new_cache = conv_input[:, -self.kernel_size + 1 :, :]
        return out, new_cache


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        if args.linear_attn_config is None:
            raise ValueError("linear_attn_config must be provided for KDA layers")
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.log(
            mx.random.uniform(
                low=1.0,
                high=16.0,
                shape=(self.num_heads,),
                dtype=mx.float32,
            )
        )
        self.dt_bias = mx.zeros((self.projection_dim,), dtype=mx.float32)

        self.o_norm = RMSNormGated(
            self.head_dim, eps=args.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        conv_state = None
        recurrent_state = None
        if cache is not None:
            conv_state = cache[0]
            recurrent_state = cache[1]

        if conv_state is None:
            zeros_q = mx.zeros(
                (B, self.conv_kernel - 1, self.projection_dim), dtype=dtype
            )
            zeros_k = mx.zeros(
                (B, self.conv_kernel - 1, self.projection_dim), dtype=dtype
            )
            zeros_v = mx.zeros(
                (B, self.conv_kernel - 1, self.projection_dim), dtype=dtype
            )
        else:
            zeros_q, zeros_k, zeros_v = conv_state

        q_conv, conv_q = self.q_conv(self.q_proj(x), zeros_q)
        k_conv, conv_k = self.k_conv(self.k_proj(x), zeros_k)
        v_conv, conv_v = self.v_conv(self.v_proj(x), zeros_v)

        if cache is not None:
            cache[0] = (conv_q, conv_k, conv_v)

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim).astype(mx.float32)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim).astype(mx.float32)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim).astype(mx.float32)

        def _l2norm(x, eps=1e-6):
            norm = mx.linalg.norm(x, axis=-1, keepdims=True)
            return x / (norm + eps)

        q = _l2norm(q)
        k = _l2norm(k)
        q = q * self.scale

        a_logits = (
            self.f_b_proj(self.f_a_proj(x))
            .reshape(B, T, self.num_heads, self.head_dim)
            .astype(mx.float32)
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads).astype(mx.float32)

        state_in = None
        if recurrent_state is not None:
            state_in = recurrent_state.astype(mx.float32)

        mask_bool: Optional[mx.array] = None
        if mask is not None:
            mask_bool = mask.astype(mx.bool_) if mask.dtype != mx.bool_ else mask
            if mask_bool.ndim == 2:
                mask_bool = mx.expand_dims(mask_bool, axis=-1)
            if mask_bool is not None and mask_bool.shape[-1] == 1:
                mask_bool = mx.broadcast_to(mask_bool, (B, T, self.num_heads))

        out, new_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=state_in,
            mask=mask_bool,
            use_kernel=True,
        )

        if cache is not None:
            cache[1] = new_state

        out = out.astype(dtype)
        gate = (
            self.g_b_proj(self.g_a_proj(x))
            .reshape(B, T, self.num_heads, self.head_dim)
            .astype(dtype)
        )
        out = self.o_norm(
            out.reshape(B, T, self.num_heads, self.head_dim), gate
        ).reshape(B, T, -1)
        return self.o_proj(out)


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.is_kda_layer(layer_idx)
        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        if (
            args.is_moe
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiSparseMoE(args)
        else:
            self.mlp = KimiMLP(args)

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
        attn_cache = None if cache is None else cache
        y = self.self_attn(self.input_layernorm(x), mask, attn_cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z


class KimiLinearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, layer_cache in zip(self.layers, cache):
            if layer.is_linear:
                mask = create_ssm_mask(h, layer_cache)
            else:
                mask = create_attention_mask(h, layer_cache)
            h = layer(h, mask=mask, cache=layer_cache)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiLinearModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(MambaCache())
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, KimiSparseMoE):
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    key = f"{src_prefix}.experts.0.{src}.weight"
                    if key in weights:
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight")
                            for i in range(self.args.num_experts)
                        ]
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                            stacked
                        )

                for name in ("gate_proj", "up_proj", "down_proj"):
                    src_key = f"{src_prefix}.shared_experts.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.shared_experts.{name}.weight"] = (
                            weights.pop(src_key)
                        )

                gate_key = f"{src_prefix}.gate.weight"
                if gate_key in weights:
                    weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                        bias_key
                    )

            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, KimiDeltaAttention):
                attn_prefix = f"{prefix}.self_attn"
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                A_key = f"{attn_prefix}.A_log"
                if A_key in weights:
                    if weights[A_key].ndim == 4:
                        weights[A_key] = mx.squeeze(weights[A_key], axis=(0, 1, 3))
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights:
                    if weights[dt_key].ndim > 1:
                        weights[dt_key] = mx.reshape(weights[dt_key], (-1,))
            elif isinstance(attn, KimiMLAAttention):
                attn_prefix = f"{prefix}.self_attn"
                proj_key = f"{attn_prefix}.kv_a_proj_with_mqa.weight"
                if proj_key in weights:
                    weights[f"{attn_prefix}.kv_a_proj.weight"] = weights.pop(proj_key)
                norm_key = f"{attn_prefix}.kv_a_layernorm.weight"
                if norm_key in weights:
                    weights[f"{attn_prefix}.kv_a_norm.weight"] = weights.pop(norm_key)

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
