# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma3 import checkpoint
from mlx_lm.models.gemma3.config import ModelArgs
from mlx_lm.models.gemma3_text.model import Gemma3TextModel, make_cache

__all__ = [
    "Gemma3Model",
    "Model",
]


class Gemma3Model(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Gemma3TextModel(args.text)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma3Model(args)
        self.lm_head = nn.Linear(
            args.text.hidden_size, args.text.vocab_size, bias=False
        )
        # sanitize ties it when the checkpoint has no output layer.
        self.tie_word_embeddings = False

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        language_model = self.model.language_model
        out = language_model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        return make_cache(self.args.text)
