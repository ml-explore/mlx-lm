# Copyright © 2026 Apple Inc.

from mlx_lm.models.qwen3_5.model import Model as Qwen3_5Model
from mlx_lm.models.qwen3_5_moe import checkpoint

__all__ = [
    "Model",
]


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)
