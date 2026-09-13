# Copyright © 2025 Apple Inc. Adapted for Agnes-3.0-Flash.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    head_dim: int
    parallel_ffn_intermediate_size: int = 0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    full_attention_interval: int = 4


@partial(mx.compile, shapeless=True)
def _precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


class AgnesRMSNorm(nn.Module):
    """Agnes uses (1 + weight) as the scale. Init to zero for identity."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class AgnesGatedNorm(nn.Module):
    """Gated RMSNorm: y = RMSNorm(x) * SiLU(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, hidden_states, gate):
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        return _precise_swiglu(hidden_states, gate, x)


class AgnesGlobalAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        q_width = self.num_attention_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, q_width * 2, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, kv_width, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, kv_width, bias=args.attention_bias)
        self.o_proj = nn.Linear(q_width, args.hidden_size, bias=args.attention_bias)

        self.q_norm = AgnesRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = AgnesRMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

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

        return self.o_proj(output * mx.sigmoid(gate))


class AgnesGatedDeltaNet(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False
        )
        self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = AgnesGatedNorm(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(
        self, mixed_qkvz: mx.array, mixed_ba: mx.array
    ) -> tuple:
        nk, dn, nv, dv = (
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )
        # HF layout: [all_q, all_k, all_v, all_z] flat
        key_dim_total = nk * dn
        value_dim_total = nv * dv

        q = mixed_qkvz[:, :, 0:key_dim_total].reshape(*mixed_qkvz.shape[:2], nk, dn)
        k = mixed_qkvz[:, :, key_dim_total : 2 * key_dim_total].reshape(*mixed_qkvz.shape[:2], nk, dn)
        v = mixed_qkvz[:, :, 2 * key_dim_total : 2 * key_dim_total + value_dim_total].reshape(
            *mixed_qkvz.shape[:2], nv, dv
        )
        z = mixed_qkvz[:, :, 2 * key_dim_total + value_dim_total :].reshape(
            *mixed_qkvz.shape[:2], nv, dv
        )

        b = mixed_ba[:, :, 0:nv]
        a = mixed_ba[:, :, nv:]

        return q, k, v, z, b, a

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)

        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class AgnesMLP(nn.Module):
    """SwiGLU with optional parallel branch."""

    def __init__(self, dim, hidden_dim, parallel_hidden_dim=0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        if parallel_hidden_dim > 0:
            self.parallel_gate_proj = nn.Linear(dim, parallel_hidden_dim, bias=False)
            self.parallel_down_proj = nn.Linear(parallel_hidden_dim, dim, bias=False)
            self.parallel_up_proj = nn.Linear(dim, parallel_hidden_dim, bias=False)
        else:
            self.parallel_gate_proj = None

    def __call__(self, x) -> mx.array:
        y = self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        if self.parallel_gate_proj is not None:
            y = y + self.parallel_down_proj(
                swiglu(self.parallel_gate_proj(x), self.parallel_up_proj(x))
            )
        return y


class AgnesDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = AgnesGatedDeltaNet(args)
        else:
            self.self_attn = AgnesGlobalAttention(args)

        self.input_layernorm = AgnesRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = AgnesRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = AgnesMLP(
            args.hidden_size,
            args.intermediate_size,
            args.parallel_ffn_intermediate_size,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class AgnesTextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            AgnesDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = AgnesRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AgnesTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        # Strip MTP weights
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Remap parallel FFN weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            if f"{prefix}.parallel_ffn.gate_proj.weight" in weights:
                weights[f"{prefix}.parallel_gate_proj.weight"] = weights.pop(
                    f"{prefix}.parallel_ffn.gate_proj.weight"
                )
                weights[f"{prefix}.parallel_up_proj.weight"] = weights.pop(
                    f"{prefix}.parallel_ffn.up_proj.weight"
                )
                weights[f"{prefix}.parallel_down_proj.weight"] = weights.pop(
                    f"{prefix}.parallel_ffn.down_proj.weight"
                )

        # Merge delta attention projections
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.delta_attn"
            if f"{prefix}.in_proj_qkv.weight" in weights:
                qkv = weights.pop(f"{prefix}.in_proj_qkv.weight")
                z = weights.pop(f"{prefix}.in_proj_z.weight")
                weights[f"model.layers.{l}.linear_attn.in_proj_qkvz.weight"] = (
                    mx.concatenate([qkv, z], axis=0)
                )
            if f"{prefix}.in_proj_b.weight" in weights:
                b = weights.pop(f"{prefix}.in_proj_b.weight")
                a = weights.pop(f"{prefix}.in_proj_a.weight")
                weights[f"model.layers.{l}.linear_attn.in_proj_ba.weight"] = (
                    mx.concatenate([b, a], axis=0)
                )
            if f"{prefix}.A_log" in weights:
                weights[f"model.layers.{l}.linear_attn.A_log"] = weights.pop(
                    f"{prefix}.A_log"
                )
            if f"{prefix}.dt_bias" in weights:
                weights[f"model.layers.{l}.linear_attn.dt_bias"] = weights.pop(
                    f"{prefix}.dt_bias"
                )
            if f"{prefix}.conv1d.weight" in weights:
                w = weights.pop(f"{prefix}.conv1d.weight")
                if w.shape[-1] != 1:
                    w = w.moveaxis(2, 1)
                weights[f"model.layers.{l}.linear_attn.conv1d.weight"] = w
            if f"{prefix}.out_proj.weight" in weights:
                weights[f"model.layers.{l}.linear_attn.out_proj.weight"] = (
                    weights.pop(f"{prefix}.out_proj.weight")
                )
            if f"{prefix}.norm.weight" in weights:
                weights[f"model.layers.{l}.linear_attn.norm.weight"] = weights.pop(
                    f"{prefix}.norm.weight"
                )

        # Remap global attention
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.global_attn"
            if f"{prefix}.q_proj.weight" in weights:
                weights[f"model.layers.{l}.self_attn.q_proj.weight"] = weights.pop(
                    f"{prefix}.q_proj.weight"
                )
            if f"{prefix}.k_proj.weight" in weights:
                weights[f"model.layers.{l}.self_attn.k_proj.weight"] = weights.pop(
                    f"{prefix}.k_proj.weight"
                )
            if f"{prefix}.v_proj.weight" in weights:
                weights[f"model.layers.{l}.self_attn.v_proj.weight"] = weights.pop(
                    f"{prefix}.v_proj.weight"
                )
            if f"{prefix}.o_proj.weight" in weights:
                weights[f"model.layers.{l}.self_attn.o_proj.weight"] = weights.pop(
                    f"{prefix}.o_proj.weight"
                )
            if f"{prefix}.q_norm.weight" in weights:
                weights[f"model.layers.{l}.self_attn.q_norm.weight"] = weights.pop(
                    f"{prefix}.q_norm.weight"
                )
            if f"{prefix}.k_norm.weight" in weights:
                weights[f"model.layers.{l}.self_attn.k_norm.weight"] = weights.pop(
                    f"{prefix}.k_norm.weight"
                )

        # One-centered norms: add 1 to HF weights
        one_centered_norms = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in list(weights.items()):
            if any(k.endswith(sfx) for sfx in one_centered_norms):
                if v.ndim == 1:
                    weights[k] = v + 1.0

        return weights
