# Copyright © 2026 Apple Inc.

from mlx_lm.models.deepseek_v3.model import Model as DeepseekV3Model
from mlx_lm.models.sarvam_mla import checkpoint
from mlx_lm.models.sarvam_mla.config import ModelArgs

__all__ = [
    "Model",
]


class Model(DeepseekV3Model):
    """Sarvam MLA using the DeepSeek-V3 attention and MoE implementation."""

    def __init__(self, config: ModelArgs):
        super().__init__(config)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)
