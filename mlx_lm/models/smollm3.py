# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import llama


@dataclass
class ModelArgs(llama.ModelArgs):
    model_type: str
    no_rope_layer_interval: int = 4
    no_rope_layers: Optional[list[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
        elif len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError("`no_rope_layers` length mismatch")


class NoPE(nn.Module):
    """No-op used to disable rotary embeddings in selected layers."""

    def __call__(self, x, offset: int = 0):
        return x


class Model(nn.Module):
    """Wrapper around Llama that respects NoPE layers in SmolLM-3."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type: str = args.model_type

        self.model = llama.LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        for idx, use_rope in enumerate(args.no_rope_layers):
            if not use_rope:
                self.model.layers[idx].self_attn.rope = NoPE()

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, mask, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict):
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
