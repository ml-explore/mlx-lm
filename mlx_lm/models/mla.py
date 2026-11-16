# Copyright © 2026 Apple Inc.

"""Moved to ``mlx_lm.layers.mla``; this module re-exports it."""

from mlx_lm.layers.mla import MultiLinear, QuantizedMultiLinear

__all__ = [
    "MultiLinear",
    "QuantizedMultiLinear",
]
