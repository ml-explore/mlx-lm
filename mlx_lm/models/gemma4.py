# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import gemma4_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4"
    text_config: dict = None
    vocab_size: int = 262144

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        self.text_config["vocab_size"] = self.vocab_size
        self.text_config["num_attention_heads"] = self.text_config.get(
            "num_attention_heads", 8
        )
        self.text_config["num_key_value_heads"] = self.text_config.get(
            "num_key_value_heads", 1
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            starts_w_model = k.startswith("model.")

            k = k.removeprefix("model.")
            if k.startswith(
                (
                    "vision_tower",
                    "multi_modal_projector",
                    "audio_tower",
                    "embed_audio",
                    "embed_vision",
                )
            ):
                continue

            if not starts_w_model:
                new_weights[k] = v
                continue

            if k.startswith("language_model"):
                k = k.replace("language_model.", "language_model.model.")

            new_weights[k] = v

        return self.language_model.sanitize(new_weights)

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    def make_cache(self):
        return self.language_model.make_cache()
