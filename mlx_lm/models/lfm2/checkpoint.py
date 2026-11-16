# Copyright © 2025 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    sanitized_weights = {}
    for name, param in weights.items():
        if "conv.weight" in name:
            if param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)

        sanitized_weights[name] = param
    return sanitized_weights
