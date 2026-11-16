# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


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
