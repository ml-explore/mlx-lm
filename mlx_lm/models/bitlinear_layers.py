# Copyright © 2025 Apple Inc.

"""Moved to ``mlx_lm.layers.bitlinear``; this module re-exports it."""

from mlx_lm.layers.bitlinear import BitLinear, bitnet_quantize, make_bitlinear_kernel

__all__ = [
    "BitLinear",
    "bitnet_quantize",
    "make_bitlinear_kernel",
]
