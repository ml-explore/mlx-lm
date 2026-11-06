# Copyright © 2026 Apple Inc.

"""Collectives that every rank calls in the same order; no group, no effect."""

import pickle
from typing import Any

import mlx.core as mx

__all__ = ["all_max", "all_min", "share", "sync_seed"]


def share(group: mx.distributed.Group | None, obj: Any) -> Any:
    """Rank 0's object, on every rank: its size, then its bytes, in two sums."""
    if group is None:
        return obj

    data = pickle.dumps(obj) if group.rank() == 0 else b""
    size = mx.distributed.all_sum(len(data), group=group).item()

    if group.rank() == 0:
        buffer = mx.array(data)
    else:
        buffer = mx.zeros(size, dtype=mx.uint8)
    buffer = mx.distributed.all_sum(buffer, group=group)
    # A collective runs only when evaluated, and every rank must run it.
    mx.eval(buffer)

    if group.rank() == 0:
        return obj
    return pickle.loads(memoryview(buffer))


def all_max(group: mx.distributed.Group | None, value: int) -> int:
    """The largest value of all ranks."""
    if group is None:
        return value

    array = mx.array(value, dtype=mx.int64)
    return mx.distributed.all_max(array, group=group).item()


def all_min(group: mx.distributed.Group | None, value: int) -> int:
    """The smallest value of all ranks."""
    if group is None:
        return value

    array = mx.array(value, dtype=mx.int64)
    return mx.distributed.all_min(array, group=group).item()


def sync_seed(group: mx.distributed.Group | None) -> None:
    """Seed every rank the same, so that all sample the same tokens."""
    if group is None:
        return

    seed = mx.distributed.all_sum(mx.random.state[0], group=group)
    mx.random.seed(seed.view(mx.uint64).item())
