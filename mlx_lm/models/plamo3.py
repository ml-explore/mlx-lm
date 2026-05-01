# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from types import MethodType
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, QuantizedKVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo3"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 2048
    window_size: int = 2048
    sliding_window: Optional[int] = None
    sliding_window_pattern: int = 8
    rope_theta: float = 1_000_000
    rope_local_theta: float = 10_000
    intermediate_size: int = 13312
    vocab_size: int = 32000
    image_token_id: Optional[int] = None
    image_feature_size: Optional[int] = None
    image_proj_type: str = "linear"
    linear_type: str = "normal"

    def __post_init__(self):
        if self.sliding_window is not None:
            self.window_size = self.sliding_window


def is_full_attention(args: ModelArgs, layer_idx: int) -> bool:
    return not bool((layer_idx + 1) % args.sliding_window_pattern)


def dequantize_cache_state(state: Any, cache: Any) -> mx.array:
    if hasattr(cache, "bits"):
        return mx.dequantize(
            *state, group_size=cache.group_size, bits=cache.bits
        )
    return state


def inverse_rope(
    x: mx.array,
    dims: int,
    *,
    base: float,
    traditional: bool = False,
) -> mx.array:
    if dims <= 0:
        return x

    x_rot = x[..., :dims]
    x_pass = x[..., dims:]
    positions = mx.arange(x.shape[-2], dtype=mx.float32)
    freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    angles = positions[:, None] / freqs[None, :]
    cos = mx.cos(angles).astype(x.dtype)[None, None, :, :]
    sin = mx.sin(angles).astype(x.dtype)[None, None, :, :]

    if traditional:
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        y_even = x_even * cos + x_odd * sin
        y_odd = x_odd * cos - x_even * sin
        y = mx.stack([y_even, y_odd], axis=-1).reshape(*x_rot.shape)
    else:
        half = dims // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        y = mx.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=-1)

    if x_pass.shape[-1] == 0:
        return y
    return mx.concatenate([y, x_pass], axis=-1)


def plamo3_full_kv_cache_to_quantized(
    cache,
    group_size: int = 64,
    bits: int = 4,
) -> QuantizedKVCache:
    quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
    quant_cache.offset = cache.offset
    quant_cache.plamo3_cache_unrotated_keys = True
    if cache.keys is not None:
        keys, values = cache.state
        if not getattr(cache, "plamo3_cache_unrotated_keys", False):
            keys = inverse_rope(
                keys,
                cache.plamo3_rope_dim,
                base=cache.plamo3_rope_base,
            )
        quant_cache.keys = mx.quantize(keys, group_size=group_size, bits=bits)
        quant_cache.values = mx.quantize(values, group_size=group_size, bits=bits)
    return quant_cache


def plamo3_sliding_kv_cache_to_quantized(
    cache,
    group_size: int = 64,
    bits: int = 4,
):
    return cache


def prepare_plamo3_cache(config: ModelArgs, layer: Any, cache: Any) -> None:
    if cache is None or hasattr(cache, "bits"):
        return
    if layer.full_attn and isinstance(cache, KVCache):
        cache.plamo3_rope_dim = config.head_dim
        cache.plamo3_rope_base = config.rope_theta
        cache.to_quantized = MethodType(plamo3_full_kv_cache_to_quantized, cache)
    elif not layer.full_attn and isinstance(cache, RotatingKVCache):
        cache.to_quantized = MethodType(plamo3_sliding_kv_cache_to_quantized, cache)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.variance_epsilon = eps
        self.offset = offset

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            hidden_states, self.weight + self.offset, self.variance_epsilon
        )


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        head_dim = config.head_dim
        self.scale = head_dim**-0.5

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        assert self.q_num_heads % self.k_num_heads == 0

        self.q_proj_dim = self.q_num_heads * self.qk_dim
        self.k_proj_dim = self.k_num_heads * self.qk_dim
        self.v_proj_dim = self.v_num_heads * self.v_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_proj_dim + self.k_proj_dim + self.v_proj_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.qk_dim, eps=config.rms_norm_eps, offset=1.0)
        self.k_norm = RMSNorm(self.qk_dim, eps=config.rms_norm_eps, offset=1.0)

        self.full_attn = is_full_attention(config, layer_idx)
        rope_base = config.rope_theta if self.full_attn else config.rope_local_theta
        self.rope = nn.RoPE(self.qk_dim, traditional=False, base=rope_base)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        queries, keys, values = mx.split(
            qkv,
            [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim],
            axis=-1,
        )

        queries = queries.reshape(B, L, self.q_num_heads, self.qk_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.k_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.v_num_heads, self.v_dim).transpose(
            0, 2, 1, 3
        )

        attn_dtype = queries.dtype
        queries = self.q_norm(queries).astype(attn_dtype)
        keys = self.k_norm(keys).astype(attn_dtype)
        attention_cache = cache

        if self.full_attn:
            if cache is not None and (
                getattr(cache, "plamo3_cache_unrotated_keys", False)
                or hasattr(cache, "bits")
            ):
                offset = cache.offset
                keys, values = cache.update_and_fetch(keys, values)
                keys = self.rope(dequantize_cache_state(keys, cache))
                values = dequantize_cache_state(values, cache)
                queries = self.rope(queries, offset=offset)
                if hasattr(cache, "bits"):
                    attention_cache = None
            else:
                offset = cache.offset if cache is not None else 0
                queries = self.rope(queries, offset=offset)
                keys = self.rope(keys, offset=offset)
                if cache is not None:
                    keys, values = cache.update_and_fetch(keys, values)
        else:
            # Sliding layers keep unrotated keys in the rotating cache and
            # reapply RoPE over the visible KV window to reset local positions.
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            query_offset = keys.shape[-2] - queries.shape[-2]
            queries = self.rope(queries, offset=query_offset)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=attention_cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            B, L, self.q_num_heads * self.v_dim
        )
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, value = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, value))


class Plamo3DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.full_attn = is_full_attention(config, layer_idx)
        self.mixer = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.pre_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)
        hidden_states_sa = self.mixer(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + self.post_mixer_norm(hidden_states_sa)

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states)
        return residual + self.post_mlp_norm(hidden_states_mlp)


class Plamo3Decoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.layers = [
            Plamo3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.full_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.full_attn), 0
        )
        self.swa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.full_attn), None
        )

    def __call__(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            prepare_plamo3_cache(self.config, layer, c)

        full_mask = create_attention_mask(x, cache[self.full_idx])
        sliding_window_mask = None
        if self.swa_idx is not None:
            sliding_window_mask = create_attention_mask(
                x,
                cache[self.swa_idx],
                window_size=self.window_size + 1,
            )

        for layer, c in zip(self.layers, cache):
            mask = full_mask if layer.full_attn else sliding_window_mask
            x = layer(x, mask=mask, cache=c)
        return x


class Plamo3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = Plamo3Decoder(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        h = self.layers(h, cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Plamo3Model(config)
        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, self.vocab_size, bias=False
            )

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.full_attn:
                c = KVCache()
            else:
                # The HF sliding mask includes the current token plus
                # window_size previous tokens.
                c = RotatingKVCache(max_size=self.config.window_size + 1)
            prepare_plamo3_cache(self.config, layer, c)
            caches.append(c)
        return caches

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        outputs = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(outputs)
        return self.lm_head(outputs)

    @property
    def layers(self):
        return self.model.layers.layers
