# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import lfm2
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    def __post_init__(self):
        self.text_config["tie_word_embeddings"] = False
        self.text_config["full_attn_idxs"] = [
            i
            for i, layer_type in enumerate(self.text_config["layer_types"])
            if layer_type == "full_attention"
        ]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = lfm2.Model(lfm2.ModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, mask=mask, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        weights.pop("vision_tower", None)
        weights.pop("multi_modal_projector", None)
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()
