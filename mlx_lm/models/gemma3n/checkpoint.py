# Copyright © 2025 Apple Inc.

from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = tree_unflatten(list(weights.items()))
    for k in ["vision_tower", "audio_tower", "embed_audio", "embed_vision"]:
        weights["model"].pop(k, None)
    return dict(tree_flatten(weights))
