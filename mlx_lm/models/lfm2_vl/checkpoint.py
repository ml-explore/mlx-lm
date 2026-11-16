# Copyright © 2025 Apple Inc.

from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = tree_unflatten(list(weights.items()))
    weights.pop("vision_tower", None)
    weights.pop("multi_modal_projector", None)
    return dict(tree_flatten(weights))
