# Copyright © 2023 Apple Inc.

"""Moved to ``mlx_lm.layers.switch``; this module re-exports it."""

from mlx_lm.layers.switch import (
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
)

__all__ = [
    "QuantizedSwitchLinear",
    "SwiGLU",
    "SwitchGLU",
    "SwitchLinear",
    "SwitchMLP",
]
