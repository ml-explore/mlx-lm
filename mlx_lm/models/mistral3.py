# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import llama, ministral3
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
        if args.text_config.get("model_type") == "ministral3":
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
        # transformers >= 5 checkpoints nest everything under a top-level
        # "model." (model.language_model.*, model.vision_tower.*) with
        # lm_head.weight at the root. Remap to the older layout
        # (language_model.model.*, language_model.lm_head.*) expected below.
        if any(k.startswith("model.language_model.") for k in weights):
            remapped = {}
            for k, v in weights.items():
                if k.startswith("model.language_model."):
                    remapped[
                        "language_model.model." + k[len("model.language_model.") :]
                    ] = v
                elif k.startswith(
                    ("model.vision_tower.", "model.multi_modal_projector.")
                ):
                    continue
                elif k.startswith("lm_head."):
                    remapped["language_model." + k] = v
                else:
                    remapped[k] = v
            weights = remapped
        weights = tree_unflatten(list(weights.items()))
        weights.pop("vision_tower", None)
        weights.pop("multi_modal_projector", None)
        lm_weights = dict(tree_flatten(weights["language_model"]))
        weights["language_model"] = self.language_model.sanitize(lm_weights)
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.language_model.model.layers
