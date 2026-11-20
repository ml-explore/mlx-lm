# Copyright © 2025 Apple Inc.

from typing import Any

__all__ = [
    "sanitize",
]


def sanitize(model, weights: dict[Any, Any]) -> dict[Any, Any]:
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
    return weights
