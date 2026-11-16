# Copyright © 2025 Apple Inc.

from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = tree_unflatten(list(weights.items()))
    weights.pop("visual", None)
    # Newer HF checkpoints nest the language model under
    # ``model.language_model.*`` and keep ``lm_head`` at the top level.
    if language_model := weights.get("model", {}).get("language_model"):
        lm_head = weights["lm_head"]
    else:
        language_model = weights["language_model"]["model"]
        lm_head = weights["language_model"]["lm_head"]

    weights = {"language_model": {"model": language_model, "lm_head": lm_head}}
    weights = dict(tree_flatten(weights))

    for l in range(model.language_model.args.num_hidden_layers):
        prefix = f"language_model.model.layers.{l}.mlp"
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        if gate_up_key in weights:
            gate_up = weights.pop(gate_up_key)
            mid = gate_up.shape[-1] // 2
            weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                ..., :mid
            ].swapaxes(-2, -1)
            weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                ..., mid:
            ].swapaxes(-2, -1)
            weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                f"{prefix}.experts.down_proj"
            ).swapaxes(-2, -1)

    return weights
