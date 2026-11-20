# Copyright © 2026 Apple Inc.

"""Muse Spark.

https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4
"""

from mlx_lm.models.muse_spark.config import TOKENIZER_DEFAULTS, ModelArgs
from mlx_lm.models.muse_spark.model import Model

__all__ = [
    "Model",
    "ModelArgs",
    "TOKENIZER_DEFAULTS",
]
