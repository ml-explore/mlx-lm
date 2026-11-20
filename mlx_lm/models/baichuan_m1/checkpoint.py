# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights: dict) -> dict:
    is_quantized = "lm_head.scales" in weights
    if not is_quantized and "lm_head.weight" in weights:
        w = weights["lm_head.weight"]
        dtype = w.dtype
        w = w.astype(mx.float32)
        norm = mx.linalg.norm(w, axis=-1, keepdims=True)
        w = (w / (norm + 1e-7)).astype(dtype)
        weights["lm_head.weight"] = w
    return weights
