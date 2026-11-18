# Copyright © 2026 Apple Inc.

import mlx.core as mx

from mlx_lm.models.checkpoint import from_text_layout

__all__ = [
    "sanitize",
]

# The norms that Hugging Face stores as an offset from 1, and this model as 1 + weight.
_OFFSET_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "language_model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _upgrade_mlx_layout(weights):
    """A checkpoint that mlx-lm or mlx-vlm converted with the language model under
    language_model."""
    upgraded = {}
    for k, v in weights.items():
        if k.startswith("vision_tower."):
            continue
        if k.startswith("language_model.model."):
            k = "model.language_model." + k.removeprefix("language_model.model.")
        else:
            k = k.removeprefix("language_model.")
        upgraded[k] = v
    return upgraded


def sanitize(model, weights):
    if any(k.startswith("language_model.") for k in weights):
        weights = _upgrade_mlx_layout(weights)
    # mlx-lm does not run the vision encoder or the multi-token prediction head yet.
    weights = {
        k: v for k, v in weights.items() if not k.startswith(("model.visual.", "mtp."))
    }
    # Text-only checkpoints have the text model's names.
    weights = {from_text_layout(k): v for k, v in weights.items()}
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    # Hugging Face stores conv1d channels-first; MLX conversions already have this
    # model's format.
    from_hugging_face = any(
        k.endswith("conv1d.weight") and v.shape[-1] != 1 for k, v in weights.items()
    )
    for k in list(weights):
        v = weights[k]
        if k.endswith("conv1d.weight") and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        elif from_hugging_face and k.endswith(_OFFSET_NORMS):
            weights[k] = v + 1.0
        elif k.endswith(".experts.gate_up_proj"):
            gate, up = mx.split(weights.pop(k), 2, axis=-2)
            switch_mlp = k.removesuffix(".experts.gate_up_proj") + ".switch_mlp"
            weights[f"{switch_mlp}.gate_proj.weight"] = gate
            weights[f"{switch_mlp}.up_proj.weight"] = up
        elif k.endswith(".experts.down_proj"):
            switch_mlp = k.removesuffix(".experts.down_proj") + ".switch_mlp"
            weights[f"{switch_mlp}.down_proj.weight"] = weights.pop(k)
    return weights
