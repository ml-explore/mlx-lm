# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import llama
from mlx_lm.models.smollm3 import checkpoint
from mlx_lm.models.smollm3.config import ModelArgs
from mlx_lm.models.smollm3.layers import NoPE

__all__ = [
    "Model",
]


class Model(nn.Module):
    """Wrapper around Llama that respects NoPE layers in SmolLM-3."""

    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type: str = args.model_type

        self.model = llama.model.LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        for idx, use_rope in enumerate(args.no_rope_layers):
            if not use_rope:
                self.model.layers[idx].self_attn.rope = NoPE()

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers
