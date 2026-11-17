# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]

# The norms whose weight is an offset from 1.
_ZERO_CENTERED_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "language_model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _upgrade_mlx_layout(weights):
    """A checkpoint that mlx-lm or mlx-vlm converted with the language model under
    language_model, the experts under switch_mlp and the norms storing 1 + weight."""
    upgraded = {}
    for k, v in weights.items():
        if k.startswith("vision_tower."):
            continue
        if k.startswith("language_model.model."):
            k = "model.language_model." + k.removeprefix("language_model.model.")
        else:
            k = k.removeprefix("language_model.")
        k = k.replace(".switch_mlp.", ".experts.")
        upgraded[k] = v - 1.0 if k.endswith(_ZERO_CENTERED_NORMS) else v
    return upgraded


def sanitize(model, weights):
    if any(k.startswith("language_model.") for k in weights):
        weights = _upgrade_mlx_layout(weights)
    # mlx-lm does not run the vision encoder or the multi-token prediction head yet.
    weights = {
        k: v for k, v in weights.items() if not k.startswith(("model.visual.", "mtp."))
    }
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for k in list(weights):
        v = weights[k]
        if k.endswith("conv1d.weight") and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        elif k.endswith(".experts.gate_up_proj"):
            gate, up = mx.split(weights.pop(k), 2, axis=-2)
            experts = k.removesuffix(".gate_up_proj")
            weights[f"{experts}.gate_proj.weight"] = gate
            weights[f"{experts}.up_proj.weight"] = up
        elif k.endswith(".experts.down_proj"):
            weights[f"{k}.weight"] = weights.pop(k)
    return weights
