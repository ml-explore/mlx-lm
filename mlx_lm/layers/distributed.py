# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "gather_last_axis",
]


def gather_last_axis(x: mx.array, group: mx.distributed.Group) -> mx.array:
    """Gather the last axis of ``x``, which is split across ``group``."""
    n, size = group.size(), x.shape[-1]
    parts = mx.distributed.all_gather(x.reshape(-1, size), group=group)
    # (ranks * rows, size) to (rows, ranks * size), in rank order.
    parts = parts.reshape(n, -1, size).transpose(1, 0, 2)
    return parts.reshape(*x.shape[:-1], n * size)
