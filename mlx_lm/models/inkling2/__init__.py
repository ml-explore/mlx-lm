# Copyright © 2026 Apple Inc.

"""Inkling 2.

https://huggingface.co/thinkingmachines/Inkling-2-Small
"""

from mlx_lm.models.inkling2.config import ModelArgs
from mlx_lm.models.inkling2.model import Model

__all__ = [
    "Model",
    "ModelArgs",
]
