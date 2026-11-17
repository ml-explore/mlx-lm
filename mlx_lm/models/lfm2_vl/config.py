# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.lfm2.config import ModelArgs as TextModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    def __post_init__(self):
        self.text = TextModelArgs.from_dict(self.text_config)
