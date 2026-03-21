# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import llama, ministral3, mistral4
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    def __post_init__(self):
        if "tie_word_embeddings" not in self.text_config:
            self.text_config["tie_word_embeddings"] = False


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Route based on architecture structure, not model_type string.
        # MoE models (Mistral Small 4) have n_routed_experts in config.
        # Dense models (Ministral 3B/8B/14B) use ministral3 or llama paths.
        if args.text_config.get("n_routed_experts") is not None:
            self.language_model = mistral4.Model(
                mistral4.ModelArgs.from_dict(args.text_config)
            )
        elif args.text_config.get("model_type") == "ministral3":
            self.language_model = ministral3.Model(
                ministral3.ModelArgs.from_dict(args.text_config)
            )
        else:
            self.language_model = llama.Model(
                llama.ModelArgs.from_dict(args.text_config)
            )

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
        weights.pop("multi_modal_projector", None)
        lm_weights = dict(tree_flatten(weights["language_model"]))
        weights["language_model"] = self.language_model.sanitize(lm_weights)
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.language_model.model.layers
