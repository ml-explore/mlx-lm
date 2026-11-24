# Copyright © 2025 Apple Inc.

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.gemma3n import checkpoint
from mlx_lm.models.gemma3n.config import ModelArgs, TextConfig
from mlx_lm.models.gemma3n.layers import Gemma3nDecoderLayer

__all__ = [
    "Gemma3n",
    "LanguageModel",
    "Model",
    "logit_softcap",
]


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    out = mx.tanh(x / softcap)
    out = out * softcap
    return out


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.vocab_size = config.vocab_size
        self.vocab_size_per_layer_input = config.vocab_size_per_layer_input
        self.num_hidden_layers = config.num_hidden_layers
        self.final_logit_softcapping = config.final_logit_softcapping
        self.first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Gemma3nDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                is_kv_shared_layer=layer_idx >= self.first_kv_shared_layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )

        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = nn.RMSNorm(
            dims=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )

        self.altup_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, self.config.altup_num_inputs)
        ]

        self.altup_unembed_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, self.config.altup_num_inputs)
        ]

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.first_sliding_idx = config.layer_types.index("sliding_attention")
        self.first_full_idx = config.layer_types.index("full_attention")
        self.sliding_window = config.sliding_window

        concrete_layers = config.layer_types[: self.first_kv_shared_layer_idx]
        shared_full_idx = (
            len(concrete_layers) - 1 - concrete_layers[::-1].index("full_attention")
        )
        shared_sliding_idx = (
            len(concrete_layers) - 1 - concrete_layers[::-1].index("sliding_attention")
        )

        self.layer_idx_to_cache_idx = []
        for i, layer_type in enumerate(self.config.layer_types):
            if i < self.first_kv_shared_layer_idx:
                self.layer_idx_to_cache_idx.append(i)
            else:
                if layer_type == "full_attention":
                    self.layer_idx_to_cache_idx.append(shared_full_idx)
                elif layer_type == "sliding_attention":
                    self.layer_idx_to_cache_idx.append(shared_sliding_idx)
                else:
                    raise NotImplementedError(f"Unknown layer type: {layer_type}")

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: mx.array = None,
    ):
        if input_embeddings is None:
            h = self.embed_tokens(inputs) * (self.hidden_size**0.5)
        else:
            h = input_embeddings

        per_layer_inputs = self.get_per_layer_inputs(inputs)
        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(
            h,
            cache[self.first_full_idx],
        )
        sliding_window_mask = create_attention_mask(
            h,
            cache[self.first_sliding_idx],
            window_size=self.sliding_window,
        )
        h0 = h

        # Expand hidden_states to support per-layer inputs
        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5

        h_list = [h0]
        h_list.extend([proj(h0) for proj in self.altup_projections])
        h = mx.stack(h_list, axis=0)
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h[1:] = h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))
        # One position offset for the whole forward pass, so KV-shared layers
        # do not read a cache their owner has already advanced.
        offset = cache[0].offset if cache is not None and cache[0] is not None else 0
        # Keep each layer's KV for the KV-shared layers.
        kvs = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = self.config.layer_types[i] == "full_attention"

            if is_global:
                mask = global_mask
            else:
                mask = sliding_window_mask

            cache_idx = self.layer_idx_to_cache_idx[i]
            h, kvs[i] = layer(
                h,
                mask,
                cache[cache_idx],
                per_layer_input,
                offset=offset,
                shared_kv=kvs[cache_idx],
            )

        # Per-layer inputs to single output
        target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5
        for i, proj in enumerate(self.altup_unembed_projections):
            h[i + 1] = proj(h[i + 1])
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h[1:] = h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))

        h = mx.mean(h, axis=0)

        out = self.norm(h)
        out = self.embed_tokens.as_linear(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return out

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        per_layer_inputs_mask = input_ids < self.vocab_size_per_layer_input
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens) * (
            self.hidden_size_per_layer_input**0.5
        )
        return result.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: mx.array,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * (
            self.hidden_size**-0.5
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * (2.0**-0.5)

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types[: self.first_kv_shared_layer_idx]:
            if layer_type == "full_attention":
                caches.append(KVCache())
            elif layer_type == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                )
            else:
                raise NotImplementedError(f"Unknown layer type: {layer_type}")
        return caches


class Gemma3n(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = LanguageModel(TextConfig.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def make_cache(self):
        return self.language_model.make_cache()


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = Gemma3n(args)
        self.model_type = args.model_type

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.model(inputs, cache=cache, input_embeddings=input_embeddings)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        return self.model.make_cache()
