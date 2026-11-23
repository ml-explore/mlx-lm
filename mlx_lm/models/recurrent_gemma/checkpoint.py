# Copyright © 2023 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for k, v in weights.items():
        if "conv_1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
    if "lm_head.weight" not in weights:
        model.pop("lm_head")
    return weights
