# Copyright © 2025 Apple Inc.

__all__ = [
    "sanitize",
    "sanitize_language_model",
]


def sanitize_language_model(language_model, weights, prefix):
    """Sanitize the language model's weights, whose names start with prefix."""
    sanitized_weights = {}
    for name, param in weights.items():
        if "conv.weight" in name:
            if param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)

        sanitized_weights[name] = param
    return sanitized_weights


def sanitize(model, weights):
    return sanitize_language_model(model.model, weights, prefix="model.")
