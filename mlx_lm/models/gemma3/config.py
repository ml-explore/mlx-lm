# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.gemma3_text.config import ModelArgs as TextModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vocab_size: int = 262208

    def __post_init__(self):
        # Gemma 3's defaults, which transformers leaves out of the text_config
        # it saves. The vocabulary size is the model's.
        text_config = {
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            **self.text_config,
            "vocab_size": self.vocab_size,
        }
        self.text = TextModelArgs.from_dict(text_config)
