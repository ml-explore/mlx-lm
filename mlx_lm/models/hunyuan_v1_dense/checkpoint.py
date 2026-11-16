# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
