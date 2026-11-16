# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.muse_glimmer import checkpoint
from mlx_lm.models.muse_glimmer.config import ModelArgs
from mlx_lm.models.muse_glimmer.layers import DecoderLayer

__all__ = [
    "Model",
    "MuseGlimmerModel",
]


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_eps = args.rms_norm_eps
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.window = args.sliding_window

    def _masks(self, h, cache):
        made, out = {}, []
        for layer, c in zip(self.layers, cache):
            t = layer.self_attn.layer_type
            if t not in made:
                if t == "sliding_attention":
                    made[t] = create_attention_mask(h, c, window_size=self.window)
                else:
                    made[t] = create_attention_mask(h, c)
            out.append(made[t])
        return out

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        # Scaleless RMSNorm on the embeddings (not Gemma's sqrt scaling).
        h = mx.fast.rms_norm(h, None, self.embed_eps)

        if cache is None:
            cache = [None] * len(self.layers)

        masks = self._masks(h, cache)
        for layer, c, mask in zip(self.layers, cache, masks):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache=cache)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        out = out * self.args.output_multiplier
        cap = self.args.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        caches = []
        for lt in self.args.layer_types:
            if lt == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches
