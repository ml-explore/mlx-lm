# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import qwen3
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = qwen3.Model(qwen3.ModelArgs.from_dict(args.text_config))

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
        weights.pop("vision_tower", None)
        weights = dict(tree_flatten(weights))

        sanitized = {}
        for key, value in weights.items():
            if not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers
