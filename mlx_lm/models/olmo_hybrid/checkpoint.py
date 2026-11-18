# Copyright © 2026 Apple Inc.

import mlx.core as mx

from mlx_lm.models.olmo_hybrid.config import ModelArgs

__all__ = [
    "sanitize",
]


def _from_transformers_save(
    args: ModelArgs, weights: dict[str, mx.array]
) -> dict[str, mx.array]:
    """Undo transformers 5's norm renames and its equal split of the q/k/v conv."""
    renamed = {}
    for k, v in weights.items():
        k = k.replace(".attention_layer_norm.", ".input_layernorm.")
        k = k.replace(".feedforward_layer_norm.", ".post_attention_layernorm.")
        renamed[k] = v
    key_dim = args.linear_num_key_heads * args.linear_key_head_dim
    for i, layer_type in enumerate(args.layer_types):
        if layer_type != "linear_attention":
            continue
        names = [
            f"model.layers.{i}.linear_attn.{p}_conv1d.weight" for p in ("q", "k", "v")
        ]
        fused = mx.concatenate([renamed[n] for n in names], axis=0)
        parts = mx.split(fused, [key_dim, 2 * key_dim], axis=0)
        for n, part in zip(names, parts):
            renamed[n] = part
    return renamed


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    if any(".feedforward_layer_norm." in k for k in weights):
        weights = _from_transformers_save(model.args, weights)
    sanitized = {}
    for k, v in weights.items():
        if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)
        sanitized[k] = v
    return sanitized
