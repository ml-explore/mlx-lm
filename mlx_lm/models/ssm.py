# Copyright © 2025 Apple Inc.

"""Moved to ``mlx_lm.layers.ssm``; this module re-exports it."""

from mlx_lm.layers.ssm import (
    compute_dt,
    make_ssm_kernel,
    segsum,
    ssm_attn,
    ssm_update,
    ssm_update_kernel,
)

__all__ = [
    "compute_dt",
    "make_ssm_kernel",
    "segsum",
    "ssm_attn",
    "ssm_update",
    "ssm_update_kernel",
]
