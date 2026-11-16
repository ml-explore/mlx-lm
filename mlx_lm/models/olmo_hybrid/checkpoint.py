# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    sanitized = {}
    for k, v in weights.items():
        if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)
        sanitized[k] = v
    return sanitized
