# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.deepseek_v3.config import ModelArgs as TextConfig

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: dict
    model_type: str = "kimi_k25"

    def __post_init__(self):
        self.text = TextConfig.from_dict(self.text_config)
