# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "ZeroCenteredRMSNorm",
]


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with the weight stored as an offset from 1."""

    def __init__(self, dims: int, *, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def _extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
