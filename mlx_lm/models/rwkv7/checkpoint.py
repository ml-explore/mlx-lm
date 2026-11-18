# Copyright © 2025 Apple Inc.

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for k in weights:
        if "k_k" in k or "k_a" in k or "g_norm" in k:
            weights[k] = weights[k].reshape(
                model.args.hidden_size // model.args.head_dim, model.args.head_dim
            )
    return weights
