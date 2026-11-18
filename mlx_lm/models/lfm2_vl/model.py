# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.lfm2.model import Lfm2Model, make_cache
from mlx_lm.models.lfm2_vl import checkpoint
from mlx_lm.models.lfm2_vl.config import ModelArgs

__all__ = [
    "Lfm2VlModel",
    "Model",
]


class Lfm2VlModel(nn.Module):
    """Puts the language model at model.language_model, as the checkpoint names it."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Lfm2Model(args.text)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2VlModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        language_model = self.model.language_model
        out = language_model(inputs, cache, input_embeddings)
        return language_model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        return make_cache(self.layers)
