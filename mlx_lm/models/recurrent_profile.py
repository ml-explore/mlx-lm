# Copyright © 2024 Apple Inc.

import json
import os
import sys
import time
from typing import Any, Callable

import mlx.core as mx


def recurrent_profile_enabled():
    value = os.environ.get("MLX_LM_PROFILE_RECURRENT")
    return value is not None and value.lower() not in {"", "0", "false", "no"}


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays(item)


def profile_recurrent_call(
    *,
    op: str,
    path: str,
    metadata: dict[str, Any],
    fn: Callable[[], Any],
):
    if not recurrent_profile_enabled():
        return fn()
    start = time.perf_counter()
    result = fn()
    arrays = list(_arrays(result))
    if arrays:
        mx.eval(*arrays)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        json.dumps(
            {
                "event": "recurrent_profile",
                "op": op,
                "path": path,
                "elapsed_ms": elapsed_ms,
                **metadata,
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    return result
