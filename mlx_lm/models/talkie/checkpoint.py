# Copyright © 2026 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Remove precomputed rotary buffers if present
    return {
        k: v
        for k, v in weights.items()
        if "rotary" not in k and not k.endswith(("cos", "sin"))
    }
