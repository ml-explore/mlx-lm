# Copyright © 2026 Apple Inc.

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
        if "tie_word_embeddings" not in self.text_config:
            self.text_config["tie_word_embeddings"] = False
