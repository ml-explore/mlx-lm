# Copyright © 2026 Apple Inc.

from mlx_lm.models.deepseek_v3.checkpoint import sanitize_language_model

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # A constant shift of all correction biases leaves routing unchanged. The
    # vLLM executor removes that redundant component when it loads the model.
    weights = {
        key: (
            value - value.mean()
            if key.endswith(".mlp.gate.e_score_correction_bias")
            else value
        )
        for key, value in weights.items()
    }
    return sanitize_language_model(model.model, weights, prefix="model.")
