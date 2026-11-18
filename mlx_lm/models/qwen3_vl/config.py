# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.qwen3.config import ModelArgs as Qwen3Args
from mlx_lm.models.qwen3_moe.config import ModelArgs as Qwen3MoeArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Hugging Face's checkpoints keep tie_word_embeddings at the top level.
        text_config = {
            "tie_word_embeddings": self.tie_word_embeddings,
            **self.text_config,
        }
        if text_config.get("num_experts", 0) > 0:
            self.text = Qwen3MoeArgs.from_dict(text_config)
        else:
            self.text = Qwen3Args.from_dict(text_config)

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)
