# Copyright © 2025 Apple Inc.

"""Moved to ``mlx_lm.layers.gated_delta``; this module re-exports it."""

from mlx_lm.layers.gated_delta import (
    compute_g,
    compute_lower_bound_g,
    gated_delta_kernel,
    gated_delta_kernel_unpacked,
    gated_delta_kernel_xtree,
    gated_delta_ops,
    gated_delta_update,
    normalize_qk,
)

__all__ = [
    "compute_g",
    "compute_lower_bound_g",
    "gated_delta_kernel",
    "gated_delta_kernel_unpacked",
    "gated_delta_kernel_xtree",
    "gated_delta_ops",
    "gated_delta_update",
    "normalize_qk",
]
