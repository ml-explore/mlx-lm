# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.gemma4_text.config import ModelArgs as TextModelArgs

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
        # Fallbacks for what transformers leaves out of the text_config it saves.
        text_config = {
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": self.vocab_size,
            **self.text_config,
        }
        self.text = TextModelArgs.from_dict(text_config)
