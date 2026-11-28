# Copyright © 2025 Apple Inc.

import mlx.nn as nn

__all__ = [
    "NoPE",
]


class NoPE(nn.Module):
    """No-op used to disable rotary embeddings in selected layers."""

    def __call__(self, x, offset: int = 0):
        return x
