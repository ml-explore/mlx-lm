# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    def __post_init__(self):
        self.text_config["tie_word_embeddings"] = False
        self.text_config["num_attention_heads"] = self.text_config.get(
            "num_attention_heads", 32
        )
