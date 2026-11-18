# Copyright © 2025 Apple Inc.

import mlx.core as mx

from mlx_lm.models.qwen2_vl.checkpoint import current_name, upgrade_mlx_layout

__all__ = [
    "sanitize",
]


def _experts_stored_input_first(weights: dict[str, mx.array], hidden_size: int) -> bool:
    """Released checkpoints store the fused experts [E, input, output], transformers 5
    [E, output, input]; gate_up_proj and down_proj are never both square."""
    for k, v in weights.items():
        if k.endswith(".experts.gate_up_proj") and v.shape[-1] != hidden_size:
            return True
        if k.endswith(".experts.down_proj") and v.shape[-2] != hidden_size:
            return True
    return False


def sanitize(model, weights):
    if any(k.startswith("language_model.") for k in weights):
        weights = upgrade_mlx_layout(weights)
    weights = {current_name(k): v for k, v in weights.items()}
    # mlx-lm does not run the vision encoder yet.
    weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    input_first = _experts_stored_input_first(weights, model.args.text.hidden_size)
    for k in list(weights):
        if k.endswith(".experts.gate_up_proj"):
            switch_mlp = k.removesuffix(".experts.gate_up_proj") + ".switch_mlp"
            gate_up = weights.pop(k)
            if input_first:
                gate_up = gate_up.swapaxes(-2, -1)
            gate, up = mx.split(gate_up, 2, axis=-2)
            weights[f"{switch_mlp}.gate_proj.weight"] = gate
            weights[f"{switch_mlp}.up_proj.weight"] = up
        elif k.endswith(".experts.down_proj"):
            switch_mlp = k.removesuffix(".experts.down_proj") + ".switch_mlp"
            down = weights.pop(k)
            if input_first:
                down = down.swapaxes(-2, -1)
            weights[f"{switch_mlp}.down_proj.weight"] = down
    return weights
