# Copyright © 2025 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if "lm_head.weight" not in weights:
        model.tie_word_embeddings = True
        model.pop("lm_head")
    return weights
