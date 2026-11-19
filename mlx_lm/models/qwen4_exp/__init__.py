# Copyright © 2026 Apple Inc.

# Qwen3.8-Flash-Next: qwen3_next with QSA sparse attention, hyper-connections
# and an n-gram embedding (PLE).

from mlx_lm.models.qwen4_exp.config import ModelArgs
from mlx_lm.models.qwen4_exp.model import Model

__all__ = [
    "Model",
    "ModelArgs",
]
