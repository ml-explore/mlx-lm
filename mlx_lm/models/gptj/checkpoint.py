# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):

    new_weights = {}
    for i in range(model.args.n_layer):
        # Remove attention bias
        if f"transformer.h.{i}.attn.masked_bias" in weights:
            del weights[f"transformer.h.{i}.attn.masked_bias"]
        if f"transformer.h.{i}.attn.bias" in weights:
            del weights[f"transformer.h.{i}.attn.bias"]

    for weight in weights:
        new_weights[weight.replace("transformer.", "model.")] = weights[weight]
    return new_weights
