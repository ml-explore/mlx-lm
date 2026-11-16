# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Remove unused precomputed rotary freqs
    return {k: v for k, v in weights.items() if "attention.rope.inv_freq" not in k}
