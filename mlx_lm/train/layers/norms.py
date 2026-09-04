# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _precise_swiglu(x, gate, y):
    gate = nn.silu(gate.astype(mx.float32))
    return (gate * y.astype(mx.float32)).astype(x.dtype)


class WeightlessRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class RMSNormGated(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dims)

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return y.astype(x.dtype)
        return _precise_swiglu(x, gate, y)
