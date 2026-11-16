# Copyright © 2025 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    return {k: v for k, v in weights.items() if "rotary_emb" not in k}
