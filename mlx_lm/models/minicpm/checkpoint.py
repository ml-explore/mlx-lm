# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    return weights
