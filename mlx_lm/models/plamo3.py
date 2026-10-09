# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope


def _make_qk_norm_rope_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="plamo3_qk_norm_rope",
        input_names=["qkv", "qw", "kw", "freqs", "offset", "eps", "mscale", "base"],
        output_names=["q", "k"],
        source="""
            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.x;
            uint start = head * 128;
            float sum = 0;
            for (uint i = 0; i < 4; ++i) {
                float x = float(qkv[start + lane * 4 + i]);
                sum += x * x;
            }
            float inv = metal::precise::rsqrt(simd_sum(sum) / 128 + eps);
            for (uint i = 0; i < 2; ++i) {
                uint d = lane + i * 32;
                T w1 = head < NQ ? qw[d] : kw[d];
                T w2 = head < NQ ? qw[d + 64] : kw[d + 64];
                // Keep the intermediate rounding of RMSNorm and YaRN.
                T x1 = T(float(qkv[start + d]) * inv) * w1;
                T x2 = T(float(qkv[start + d + 64]) * inv) * w2;
                x1 = T(x1 * T(mscale));
                x2 = T(x2 * T(mscale));
                float freq = YARN ? 1.0f / freqs[d]
                                  : metal::exp2(-float(d) / 64 * base);
                float angle = float(offset) * freq;
                float co = metal::fast::cos(angle);
                float si = metal::fast::sin(angle);
                T y1 = T(float(x1) * co - float(x2) * si);
                T y2 = T(float(x1) * si + float(x2) * co);
                if (head < NQ) {
                    q[head * 128 + d] = y1;
                    q[head * 128 + d + 64] = y2;
                } else {
                    k[(head - NQ) * 128 + d] = y1;
                    k[(head - NQ) * 128 + d + 64] = y2;
                }
            }
        """,
    )


_qk_norm_rope_kernel = _make_qk_norm_rope_kernel()


def _make_residual_norm_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="plamo3_residual_norm",
        input_names=["x", "residual", "post", "pre", "eps"],
        output_names=["h", "normalized"],
        source="""
            uint lane = thread_index_in_simdgroup;
            uint group = simdgroup_index_in_threadgroup;
            uint start = thread_position_in_threadgroup.x * 4;
            threadgroup float sums[32];
            threadgroup float inv;
            float values[4];
            for (uint i = 0; i < 4; ++i) values[i] = float(x[start + i]);
            for (uint pass = 0; pass < 2; ++pass) {
                float sum = 0;
                for (uint i = 0; i < 4; ++i) sum += values[i] * values[i];
                sum = simd_sum(sum);
                if (group == 0) sums[lane] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) sums[group] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (group == 0) {
                    sum = simd_sum(sums[lane]);
                    if (lane == 0) inv = metal::precise::rsqrt(sum / D + eps);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0; i < 4; ++i) {
                    uint d = start + i;
                    T value = T(values[i] * inv);
                    if (pass == 0) {
                        T scaled = value * post[d];
                        T added = residual[d] + scaled;
                        h[d] = added;
                        values[i] = float(added);
                    } else {
                        normalized[d] = value * pre[d];
                    }
                }
            }
        """,
    )


_residual_norm_kernel = _make_residual_norm_kernel()


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo3"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    scale_embedding: bool = False
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 2048
    window_size: int = 2048
    sliding_window: Optional[int] = None
    sliding_window_pattern: int = 8
    rope_theta: float = 1_000_000
    rope_local_theta: float = 10_000
    rope_scaling_factor: Optional[float] = None
    initial_context_length: Optional[int] = None
    intermediate_size: int = 13312
    vocab_size: int = 32000
    image_token_id: Optional[int] = None
    image_feature_size: Optional[int] = None
    image_proj_type: str = "linear"
    linear_type: str = "normal"

    def __post_init__(self):
        if self.sliding_window is not None:
            self.window_size = self.sliding_window
        if (
            self.rope_scaling_factor not in (None, 1)
            and self.initial_context_length is None
        ):
            raise ValueError("Scaled RoPE requires initial_context_length")

    @property
    def attention_window_size(self):
        # Older configs omit RoPE scaling and count only past tokens in the window.
        return self.window_size + (self.rope_scaling_factor is None)


def is_full_attention(args: ModelArgs, layer_idx: int) -> bool:
    return not bool((layer_idx + 1) % args.sliding_window_pattern)


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
        self._scale = None
        self._scale_weight = None

    @property
    def scale(self) -> mx.array:
        if self.offset == 0:
            return self.weight
        if self.training:
            return self.weight + self.offset
        if self._scale is None or self._scale_weight is not self.weight:
            self._scale = self.weight + self.offset
            self._scale_weight = self.weight
        return self._scale

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(hidden_states, self.scale, self.variance_epsilon)


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
        scaling_config = None
        if self.full_attn and config.rope_scaling_factor not in (None, 1):
            scaling_config = {
                "rope_type": "yarn",
                "factor": config.rope_scaling_factor,
                "original_max_position_embeddings": config.initial_context_length,
                "truncate": False,
            }
        self.rope = initialize_rope(
            self.qk_dim,
            base=rope_base,
            traditional=False,
            scaling_config=scaling_config,
            max_position_embeddings=config.max_position_embeddings,
        )
        self._yarn = scaling_config is not None
        self._rope_freqs = self.rope._freqs if self._yarn else mx.array([1.0])
        self._rope_base = mx.array(math.log2(rope_base))
        self._rope_mscale = mx.array(self.rope.mscale if self._yarn else 1.0)
        self._norm_eps = mx.array(config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        offset = cache.offset if cache is not None else 0
        if (
            not self.training
            and B == L == 1
            and self.qk_dim == 128
            and isinstance(offset, int)
            and _qk_norm_rope_kernel is not None
            and mx.default_device() == mx.gpu
            and qkv.dtype == self.q_norm.weight.dtype == self.k_norm.weight.dtype
        ):
            queries, keys = _qk_norm_rope_kernel(
                inputs=[
                    qkv,
                    self.q_norm.scale,
                    self.k_norm.scale,
                    self._rope_freqs,
                    mx.array(offset),
                    self._norm_eps,
                    self._rope_mscale,
                    self._rope_base,
                ],
                template=[
                    ("T", qkv.dtype),
                    ("NQ", self.q_num_heads),
                    ("YARN", self._yarn),
                ],
                grid=(32 * (self.q_num_heads + self.k_num_heads), 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (B, self.q_num_heads, L, self.qk_dim),
                    (B, self.k_num_heads, L, self.qk_dim),
                ],
                output_dtypes=[qkv.dtype, qkv.dtype],
            )
            values = qkv[..., self.q_proj_dim + self.k_proj_dim :]
        else:
            queries, keys, values = mx.split(
                qkv,
                [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim],
                axis=-1,
            )
            queries = queries.reshape(B, L, self.q_num_heads, self.qk_dim).transpose(
                0, 2, 1, 3
            )
            keys = keys.reshape(B, L, self.k_num_heads, self.qk_dim).transpose(
                0, 2, 1, 3
            )
            queries = self.rope(self.q_norm(queries), offset=offset)
            keys = self.rope(self.k_norm(keys), offset=offset)

        values = values.reshape(B, L, self.v_num_heads, self.v_dim).transpose(
            0, 2, 1, 3
        )
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
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
        self._norm_eps = mx.array(config.rms_norm_eps)
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
        if (
            not self.training
            and hidden_states.shape[:2] == (1, 1)
            and self.hidden_size % 128 == 0
            and self.hidden_size <= 4096
            and _residual_norm_kernel is not None
            and mx.default_device() == mx.gpu
            and hidden_states_sa.dtype
            == residual.dtype
            == self.post_mixer_norm.weight.dtype
            == self.pre_mlp_norm.weight.dtype
        ):
            residual, hidden_states = _residual_norm_kernel(
                inputs=[
                    hidden_states_sa,
                    residual,
                    self.post_mixer_norm.scale,
                    self.pre_mlp_norm.scale,
                    self._norm_eps,
                ],
                template=[("T", residual.dtype), ("D", self.hidden_size)],
                grid=(self.hidden_size // 4, 1, 1),
                threadgroup=(self.hidden_size // 4, 1, 1),
                output_shapes=[residual.shape, residual.shape],
                output_dtypes=[residual.dtype, residual.dtype],
            )
        else:
            residual = residual + self.post_mixer_norm(hidden_states_sa)
            hidden_states = self.pre_mlp_norm(residual)
        hidden_states_mlp = self.mlp(hidden_states)
        return residual + self.post_mlp_norm(hidden_states_mlp)


class Plamo3Decoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.window_size = config.attention_window_size
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

        full_mask = create_attention_mask(x, cache[self.full_idx])
        sliding_window_mask = None
        if self.swa_idx is not None:
            sliding_window_mask = create_attention_mask(
                x,
                cache[self.swa_idx],
                window_size=self.window_size,
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
            if self.config.scale_embedding:
                h = h * self.config.hidden_size**0.5

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
                c = RotatingKVCache(max_size=self.config.attention_window_size)
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
