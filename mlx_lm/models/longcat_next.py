# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Optional, Union

import mlx.nn as nn

from .base import BaseModelArgs
from .longcat_flash_ngram import Model as LongcatFlashNgramLM
from .longcat_flash_ngram import ModelArgs as TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextConfig, dict] = None
    text_vocab_plus_multimodal_special_token_size: int = 131125
    model_type: str = "longcat_next"

    def __post_init__(self):
        if self.text_config is None:
            raise ValueError("text_config is required")
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params):
        text_config = dict(params)
        # Ngram hashing uses text_vocab_size, not the full multimodal vocab_size
        if "text_vocab_size" in params:
            text_config["vocab_size"] = params["text_vocab_size"]
        return cls(
            text_config=text_config,
            **{
                k: v
                for k, v in params.items()
                if k in ("model_type", "text_vocab_plus_multimodal_special_token_size")
            }
        )


class Model(LongcatFlashNgramLM):
    def __init__(self, config: ModelArgs):
        super().__init__(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_vocab_plus_multimodal_special_token_size,
            bias=False,
        )

    def sanitize(self, weights):
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(
                (
                    "visual_head.",
                    "audio_head.",
                    "visual_model.",
                    "image_decoder.",
                    "image_refiner.",
                    "model.visual_tokenizer.",
                    "model.audio_tokenizer.",
                )
            )
        }
        # Truncate embed_tokens to text vocab (drop multimodal token embeddings)
        embed_key = "model.embed_tokens.weight"
        if embed_key in weights:
            weights[embed_key] = weights[embed_key][: self.args.vocab_size]
        return super().sanitize(weights)
