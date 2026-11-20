# Copyright © 2025 Apple Inc.

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.gemma4_text import checkpoint
from mlx_lm.models.gemma4_text.config import ModelArgs
from mlx_lm.models.gemma4_text.layers import DecoderLayer

__all__ = [
    "Gemma4TextModel",
    "Model",
    "logit_softcap",
    "make_cache",
    "moe_quant_predicate",
]


@partial(mx.compile, shapeless=True)
def _complete_square(x2, y2, xy):
    return x2 + mx.expand_dims(y2, -1) - 2 * xy


class Gemma4TextModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Per-layer input embeddings (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_scale = config.hidden_size**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = nn.RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        # Arrange for shared KVs
        self.previous_kvs = list(range(len(self.layers)))
        if config.num_kv_shared_layers > 0:
            N = len(self.layers)
            M = N - config.num_kv_shared_layers
            kvs_by_type = {}
            for i in range(M):
                kvs_by_type[self.layers[i].layer_type] = i
            for j in range(M, N):
                self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        # Make the initial hidden state
        if input_embeddings is None:
            input_embeddings = self.embed_tokens(inputs)
        h = input_embeddings
        h = h * self.embed_scale

        # Get the extra inputs per layer if we have per layer embeddings
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self._get_per_layer_inputs(inputs, input_embeddings)
            per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)
        if per_layer_inputs is not None:
            per_layer_inputs = [
                per_layer_inputs[:, :, i, :] for i, _ in enumerate(self.layers)
            ]
        else:
            per_layer_inputs = [None] * len(self.layers)

        # Make the kv cache list, be sure to append None for all the shared kv
        # layers
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        # Apply each layer. We save all intermediate kvs and offset and grab
        # the previous one for the shared kv layers.
        masks = self._make_masks(h, cache)
        intermediates = [(None, None)] * len(self.layers)
        for idx, (layer, c, mask, prev_idx, per_layer_input) in enumerate(
            zip(
                self.layers,
                cache,
                masks,
                self.previous_kvs,
                per_layer_inputs,
            )
        ):
            kvs, offset = intermediates[prev_idx]

            h, kvs, offset = layer(
                h,
                mask,
                c,
                per_layer_input=per_layer_input,
                shared_kv=kvs,
                offset=offset,
            )

            intermediates[idx] = (kvs, offset)

        return self.norm(h)

    def _get_per_layer_inputs(
        self,
        input_ids: Optional[mx.array],
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_ids is None:
            if input_embeddings is None:
                raise RuntimeError(
                    "input_embeddings must be provided when input_ids are omitted."
                )

            # Split the sequence dimension if this still holds too much
            # memory. 260k vocab means the distance tensor would be ~1GB
            # per 2k tokens in bf16.
            #
            # If the embedding is quantized we have to dequantize it anyway to
            # perform the match test.
            norms_embedding = self.embed_tokens.weight.square().sum(-1)
            norms_input = input_embeddings.square().sum(-1)
            distance = _complete_square(
                norms_embedding,
                norms_input,
                self.embed_tokens.as_linear(input_embeddings),
            )

            # Checks can be added if needed but they necessarily break the GPU
            # pipelining and force an eval.
            #
            #   match_counts = (distance < eps).sum(-1)
            #
            input_ids = mx.stop_gradient(mx.argmin(distance, axis=-1))

        result = self.embed_tokens_per_layer(input_ids)
        result = result * self.embed_tokens_per_layer_scale
        return mx.unflatten(
            result,
            -1,
            (self.config.num_hidden_layers, self.hidden_size_per_layer_input),
        )

    def _project_per_layer_inputs(
        self,
        input_embeddings: mx.array,
        per_layer_inputs: Optional[mx.array] = None,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(input_embeddings)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        per_layer_projection = mx.unflatten(
            per_layer_projection,
            -1,
            (self.config.num_hidden_layers, self.hidden_size_per_layer_input),
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def _make_masks(self, h, cache):
        mask = {}
        masks = []
        for l, c in zip(self.layers, cache):
            if l.layer_type not in mask:
                if l.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, c)
                elif l.layer_type == "sliding_attention":
                    mask["sliding_attention"] = create_attention_mask(
                        h, c, window_size=self.window_size
                    )
            masks.append(mask[l.layer_type])
        return masks


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


def moe_quant_predicate(path, _):
    """Keep the routers in 8 bits."""
    if path.endswith("router.proj"):
        return {"group_size": 64, "bits": 8}
    return True


def make_cache(args: ModelArgs):
    """A cache for each layer before the KV-shared ones, which reuse earlier caches."""
    first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers
    caches = []
    for i in range(first_kv_shared):
        if args.layer_types[i] == "full_attention":
            caches.append(KVCache())
        else:
            caches.append(RotatingKVCache(max_size=args.sliding_window, keep=0))
    return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        out = self.model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        return moe_quant_predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return make_cache(self.args)

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
