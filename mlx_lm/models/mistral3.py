# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import llama, ministral3, mistral4_text
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
        elif args.text_config.get("model_type") == "mistral4":
            self.language_model = mistral4_text.Model(
                mistral4_text.ModelArgs.from_dict(args.text_config)
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
        sanitized = {}
        for key, value in weights.items():
            if "vision_tower" in key or "multi_modal_projector" in key:
                continue
            if key.startswith("model."):
                key = key[len("model."):]
            sanitized[key] = value

        lm_weights = {
            k[len("language_model."):]: v
            for k, v in sanitized.items()
            if k.startswith("language_model.")
        }

        sanitized_lm = self.language_model.sanitize(lm_weights)
        return {"language_model." + k: v for k, v in sanitized_lm.items()}

    @property
    def layers(self):
        return self.language_model.model.layers
