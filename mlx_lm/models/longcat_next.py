# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

import mlx.nn as nn

from .base import BaseModelArgs
from .longcat_flash_ngram import Model as LongcatFlashNgramLM
from .longcat_flash_ngram import ModelArgs as TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "longcat_next"
    text_config: dict = None
    text_vocab_plus_multimodal_special_token_size: int = 131125

    @classmethod
    def from_dict(cls, params):
        text_config = dict(params)
        if "text_vocab_size" in params:
            text_config["vocab_size"] = params["text_vocab_size"]
        return cls(
            text_config=text_config,
            **{
                k: v
                for k, v in params.items()
                if k in ("model_type", "text_vocab_plus_multimodal_special_token_size")
            },
        )


class Model(LongcatFlashNgramLM):
    def __init__(self, config: ModelArgs):
        super().__init__(TextConfig.from_dict(config.text_config))
        self.lm_head = nn.Linear(
            self.args.hidden_size,
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
                    "model.visual_tokenizer.",
                    "model.audio_tokenizer.",
                )
            )
        }
        if "model.embed_tokens.weight" in weights:
            weights["model.embed_tokens.weight"] = weights["model.embed_tokens.weight"][
                : self.args.vocab_size
            ]
        return super().sanitize(weights)
