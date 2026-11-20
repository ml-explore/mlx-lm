# Copyright © 2026 Apple Inc.

from typing import Any

__all__ = [
    "sanitize",
]


def sanitize(model, weights: dict[Any, Any]) -> dict[Any, Any]:
    if model.config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
