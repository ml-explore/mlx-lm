# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen2.model import Qwen2Model, shard_layers
from mlx_lm.models.qwen2_vl import checkpoint
from mlx_lm.models.qwen2_vl.config import ModelArgs

__all__ = [
    "Model",
    "Qwen2VLModel",
]


class Qwen2VLModel(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Qwen2Model(args.text)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen2VLModel(args)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        language_model = self.model.language_model
        out = language_model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.text.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.language_model.layers

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        shard_layers(self.layers, group or mx.distributed.init())
