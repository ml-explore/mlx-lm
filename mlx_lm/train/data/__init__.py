# Copyright © 2026 Apple Inc.

"""Pretraining data: two sources, one packer.

:func:`load_s3` reads a corpus already shuffled offline into S3 and is what a real
run should use; :func:`load_hf` streams from the Hugging Face hub and needs no
preparation, which makes it the easy way to try a run. Both hand
:func:`iterate_batches` the same documents.
"""

from mlx_lm.train.data.batching import (
    dolma,
    get_documents,
    iterate_batches,
    prefetch,
)
from mlx_lm.train.data.hf import load_hf
from mlx_lm.train.data.s3 import load_s3

__all__ = [
    "dolma",
    "iterate_batches",
    "load_hf",
    "load_s3",
    "prefetch",
    "get_documents",
]
