# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    new_weights = {}
    for key, value in weights.items():
        if key.startswith("vision_tower") or key.startswith("model.visual"):
            continue
        if key.startswith("model.language_model"):
            key = key.replace("model.language_model", "language_model.model")
        elif key.startswith("language_model."):
            pass
        else:
            key = "language_model." + key
        new_weights[key] = value

    for l in range(model.language_model.args.num_hidden_layers):
        prefix = f"language_model.model.layers.{l}.mlp"
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        if gate_up_key in new_weights:
            gate_up = new_weights.pop(gate_up_key)
            mid = gate_up.shape[-2] // 2
            new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
            new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
            new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                f"{prefix}.experts.down_proj"
            )

    return model.language_model.sanitize(new_weights)
