# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import compute_g, gated_delta_kernel, gated_delta_ops
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    linear_allow_neg_eigval: bool
    tie_word_embeddings: bool
    attention_bias: bool
    layer_norm_epsilon: float = 1e-5
    head_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict] = None
    rope_theta: Optional[float] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", 10000.0) or 10000.0
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i % 4 == 3) else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        return normed * nn.silu(gate)


class GatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.allow_neg_eigval = args.linear_allow_neg_eigval

        self.q_proj = nn.Linear(args.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.g_proj = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

        self.q_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            groups=self.key_dim,
            padding=0,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            groups=self.key_dim,
            padding=0,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.value_dim,
            out_channels=self.value_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            groups=self.value_dim,
            padding=0,
        )

        self.A_log = mx.zeros(self.num_v_heads)
        self.dt_bias = mx.zeros(self.num_v_heads)

        self.o_norm = RMSNormGated(self.head_v_dim, eps=args.layer_norm_epsilon)

    def _apply_conv1d(
        self,
        x: mx.array,
        conv: nn.Conv1d,
        conv_state: Optional[mx.array],
        cache_slot: Optional[Any],
        slot_idx: int,
    ) -> mx.array:
        B, S, D = x.shape
        n_keep = self.conv_kernel_size - 1

        if conv_state is not None:
            padded = mx.concatenate([conv_state, x], axis=1)
        else:
            padded = mx.concatenate(
                [mx.zeros((B, n_keep, D), dtype=x.dtype), x], axis=1
            )

        if cache_slot is not None:
            cache_slot[slot_idx] = padded[:, -n_keep:, :]

        return nn.silu(conv(padded))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        conv_q_state = cache[0] if cache is not None else None
        conv_k_state = cache[1] if cache is not None else None
        conv_v_state = cache[2] if cache is not None else None

        q = self._apply_conv1d(q, self.q_conv1d, conv_q_state, cache, 0)
        k = self._apply_conv1d(k, self.k_conv1d, conv_k_state, cache, 1)
        v = self._apply_conv1d(v, self.v_conv1d, conv_v_state, cache, 2)

        q = q.reshape(B, S, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, S, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if self.num_v_heads > self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            q = mx.repeat(q, repeat, axis=2)
            k = mx.repeat(k, repeat, axis=2)

        inv_scale = self.head_k_dim ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        beta = mx.sigmoid(self.b_proj(x))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        a = self.a_proj(x)
        g = compute_g(self.A_log, a, self.dt_bias)

        state = cache[3] if (cache is not None and cache[3] is not None) else None
        if state is None:
            state = mx.zeros(
                (B, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=mx.float32,
            )

        use_kernel = (
            not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )
        if use_kernel:
            out, state = gated_delta_kernel(q, k, v, g, beta, state, mask)
        else:
            out, state = gated_delta_ops(q, k, v, g, beta, state, mask)

        if cache is not None:
            cache[3] = state
            cache.advance(S)

        gate = self.g_proj(x)
        out = out.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        out = self.o_norm(out, gate)
        out = out.reshape(B, S, -1)

        return self.o_proj(out)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        total_q_dim = args.num_attention_heads * self.head_dim
        total_kv_dim = args.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(args.hidden_size, total_q_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, total_kv_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, total_kv_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(total_q_dim, args.hidden_size, bias=args.attention_bias)

        self.q_norm = nn.RMSNorm(total_q_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(total_kv_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LinearAttentionDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.linear_attn = GatedDeltaNet(args)
        self.mlp = MLP(args)
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
        h = x + self.linear_attn(self.input_layernorm(x), mask=mask, cache=cache)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class FullAttentionDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.post_attention_layernorm(self.self_attn(x, mask=mask, cache=cache))
        out = h + self.post_feedforward_layernorm(self.mlp(h))
        return out


class OlmoHybridModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LinearAttentionDecoderLayer(args)
            if lt == "linear_attention"
            else FullAttentionDecoderLayer(args)
            for lt in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._fa_idx = next(
            i for i, lt in enumerate(args.layer_types) if lt == "full_attention"
        )
        self._lin_idx = next(
            i for i, lt in enumerate(args.layer_types) if lt == "linear_attention"
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self._fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self._lin_idx])

        for layer, c in zip(self.layers, cache):
            if isinstance(layer, FullAttentionDecoderLayer):
                h = layer(h, mask=fa_mask, cache=c)
            else:
                h = layer(h, mask=ssm_mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OlmoHybridModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for lt in self.args.layer_types:
            if lt == "linear_attention":
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            k = k.replace(".attention_layer_norm.", ".input_layernorm.")
            k = k.replace(".feedforward_layer_norm.", ".post_attention_layernorm.")
            sanitized[k] = v
        return sanitized
