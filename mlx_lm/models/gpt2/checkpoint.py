# Copyright © 2023 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    new_weights = {}
    for i in range(model.args.n_layer):
        if f"h.{i}.attn.bias" in weights:
            del weights[f"h.{i}.attn.bias"]
        if f"h.{i}.attn.c_attn.weight" in weights:
            weights[f"h.{i}.attn.c_attn.weight"] = weights[
                f"h.{i}.attn.c_attn.weight"
            ].transpose(1, 0)
        if f"h.{i}.attn.c_proj.weight" in weights:
            weights[f"h.{i}.attn.c_proj.weight"] = weights[
                f"h.{i}.attn.c_proj.weight"
            ].transpose(1, 0)
        if f"h.{i}.mlp.c_fc.weight" in weights:
            weights[f"h.{i}.mlp.c_fc.weight"] = weights[
                f"h.{i}.mlp.c_fc.weight"
            ].transpose(1, 0)
        if f"h.{i}.mlp.c_proj.weight" in weights:
            weights[f"h.{i}.mlp.c_proj.weight"] = weights[
                f"h.{i}.mlp.c_proj.weight"
            ].transpose(1, 0)
    for weight in weights:
        if not weight.startswith("model."):
            new_weights[f"model.{weight}"] = weights[weight]
        else:
            new_weights[weight] = weights[weight]
    return new_weights
