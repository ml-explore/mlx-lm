# Copyright © 2025 Apple Inc.

"""Gemma 4 text-only model for mlx-lm.

Supports Per-Layer Embeddings (PLE), KV sharing, proportional RoPE,
heterogeneous head dims, double-wide MLP, v_norm, and layer_scalar.

Key differences from Gemma 3:
  - RMSNorm uses plain weight (no +1 offset)
  - Attention scale is 1.0 (QK-norm replaces 1/sqrt(d) scaling)
  - v_norm: RMS normalization on values without learned scale
  - ProportionalRoPE: partial rotation for global attention layers
  - Per-Layer Embeddings: per-layer token embeddings gated into each layer
  - KV sharing: later layers reuse KV cache from earlier layers
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    hidden_activation: str = "gelu_pytorch_tanh"
    sliding_window: int = 512
    max_position_embeddings: int = 131072
    final_logit_softcapping: float = 30.0
    layer_types: List[str] = field(default_factory=list)
    rope_parameters: Dict = field(default_factory=dict)
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    use_double_wide_mlp: bool = True
    attention_k_eq_v: bool = False
    num_global_key_value_heads: Optional[int] = None
    enable_moe_block: bool = False
    attention_bias: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4"
    text_config: dict = field(default_factory=dict)
    vocab_size: int = 262144


class RMSNorm(nn.Module):
    """Gemma 4 RMSNorm — plain weight, no +1 offset (unlike Gemma 3)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RMSNoScale(nn.Module):
    """RMSNorm without learned scale (for v_norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class ProportionalRoPE(nn.Module):
    """RoPE with partial rotation for global attention layers.

    Only a fraction of head dims are rotated; the rest pass through unchanged.
    mx.fast.rope doesn't handle zero frequencies, so we split the tensor,
    apply RoPE only to rotated dims, and concatenate back.
    """

    def __init__(self, head_dim: int, partial_rotary_factor: float, base: float):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dims = int(partial_rotary_factor * head_dim)

        rope_angles = self.rotary_dims // 2
        self._freqs = 1.0 / (
            base ** (mx.arange(0, 2 * rope_angles, 2, dtype=mx.float32) / head_dim)
        )

    def __call__(self, x, offset=0):
        rot = x[..., : self.rotary_dims]
        passthrough = x[..., self.rotary_dims :]

        rot = mx.fast.rope(
            rot,
            self.rotary_dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )
        return mx.concatenate([rot, passthrough], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = 1.0  # QK-norm replaces 1/sqrt(d) scaling

        first_kv_shared_idx = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = (
            config.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_idx
        )

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = RMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)

        if self.is_sliding:
            rope_params = config.rope_parameters.get("sliding_attention", {})
            self.rope = nn.RoPE(
                self.head_dim,
                traditional=False,
                base=rope_params.get("rope_theta", 10000.0),
            )
        else:
            rope_params = config.rope_parameters.get("full_attention", {})
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
            rope_theta = rope_params.get("rope_theta", 1000000.0)
            if partial_rotary_factor < 1.0:
                self.rope = ProportionalRoPE(
                    self.head_dim, partial_rotary_factor, rope_theta
                )
            else:
                self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)

        offset = cache.offset if cache is not None else 0

        if self.is_kv_shared_layer and cache is not None and cache.offset > 0:
            keys, values = cache.state
        else:
            keys = self.k_proj(x)
            keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys)

            values = self.v_proj(x)
            values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            values = self.v_norm(values)

            keys = self.rope(keys, offset=offset)

            if cache is not None and not self.is_kv_shared_layer:
                keys, values = cache.update_and_fetch(keys, values)

        queries = self.rope(queries, offset=offset)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared = config.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_idx
        use_double = config.use_double_wide_mlp and is_kv_shared
        intermediate = config.intermediate_size * (2 if use_double else 1)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, config.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                config.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            h = self.per_layer_input_gate(x)
            h = nn.gelu_approx(h)
            h = h * per_layer_input
            h = self.per_layer_projection(h)
            h = self.post_per_layer_input_norm(h)
            x = residual + h

        x = x * self.layer_scalar
        return x


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.final_logit_softcapping = config.final_logit_softcapping

        self.first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
            if config.num_kv_shared_layers > 0
            else config.num_hidden_layers
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Per-Layer Embeddings — split into per-layer chunks to stay under
        # Metal's 4GB single-buffer limit
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = [
                nn.Embedding(
                    config.vocab_size_per_layer_input,
                    config.hidden_size_per_layer_input,
                )
                for _ in range(config.num_hidden_layers)
            ]
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                dims=config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )

        self.layers = [
            DecoderLayer(config=config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Cache index mapping: shared layers point to their source layer's cache
        concrete_types = config.layer_types[: self.first_kv_shared_layer_idx]
        self.layer_idx_to_cache_idx = []
        for i, lt in enumerate(config.layer_types):
            if i < self.first_kv_shared_layer_idx:
                self.layer_idx_to_cache_idx.append(i)
            else:
                idx = len(concrete_types) - 1 - concrete_types[::-1].index(lt)
                self.layer_idx_to_cache_idx.append(idx)

        self.first_sliding_idx = (
            config.layer_types.index("sliding_attention")
            if "sliding_attention" in config.layer_types
            else 0
        )
        self.first_full_idx = (
            config.layer_types.index("full_attention")
            if "full_attention" in config.layer_types
            else 0
        )
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: mx.array = None,
    ):
        if input_embeddings is None:
            h = self.embed_tokens(inputs)
            h = h * mx.array(self.hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)
        else:
            h = input_embeddings

        per_layer_inputs = None
        if self.hidden_size_per_layer_input and inputs is not None:
            per_layer_inputs = self._get_per_layer_inputs(inputs)
            per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(h, cache[self.first_full_idx])
        sliding_mask = create_attention_mask(
            h, cache[self.first_sliding_idx], window_size=self.sliding_window
        )

        for i, layer in enumerate(self.layers):
            is_global = self.config.layer_types[i] == "full_attention"
            mask = global_mask if is_global else sliding_mask
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            h = layer(h, mask, cache[self.layer_idx_to_cache_idx[i]], per_layer_input)

        h = self.norm(h)
        out = self.embed_tokens.as_linear(h)

        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)

        return out

    def _get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        mask = input_ids < self.config.vocab_size_per_layer_input
        tokens = mx.where(mask, input_ids, mx.zeros_like(input_ids))
        scale = self.hidden_size_per_layer_input**0.5
        chunks = [emb(tokens) * scale for emb in self.embed_tokens_per_layer]
        return mx.stack(chunks, axis=-2)

    def _project_per_layer_inputs(
        self, inputs_embeds: mx.array, per_layer_inputs: mx.array
    ) -> mx.array:
        per_layer_proj = self.per_layer_model_projection(inputs_embeds)
        per_layer_proj = per_layer_proj * (self.hidden_size**-0.5)
        per_layer_proj = per_layer_proj.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)
        return (per_layer_proj + per_layer_inputs) * (2.0**-0.5)

    def make_cache(self):
        caches = []
        for lt in self.config.layer_types[: self.first_kv_shared_layer_idx]:
            if lt == "full_attention":
                caches.append(KVCache())
            elif lt == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.sliding_window, keep=0))
            else:
                raise ValueError(f"Unknown layer type: {lt}")
        return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        text_cfg = args.text_config.copy()
        if "vocab_size" not in text_cfg:
            text_cfg["vocab_size"] = args.vocab_size
        self.language_model = LanguageModel(TextConfig.from_dict(text_cfg))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        for k in [
            "vision_tower",
            "audio_tower",
            "embed_vision",
            "embed_audio",
            "multi_modal_projector",
        ]:
            weights.get("model", {}).pop(k, None)
        lm_weights = weights.get("model", {}).get("language_model", weights)
        result = dict(tree_flatten({"language_model": lm_weights}))

        # Split embed_tokens_per_layer [vocab, num_layers * ple_dim]
        # into per-layer chunks to stay under Metal's 4GB buffer limit
        ple_key = "language_model.embed_tokens_per_layer.weight"
        if ple_key in result:
            big_weight = result.pop(ple_key)
            ple_dim = self.language_model.hidden_size_per_layer_input
            n_layers = self.language_model.num_hidden_layers
            for i in range(n_layers):
                chunk = big_weight[:, i * ple_dim : (i + 1) * ple_dim]
                result[f"language_model.embed_tokens_per_layer.{i}.weight"] = chunk

        return result

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
