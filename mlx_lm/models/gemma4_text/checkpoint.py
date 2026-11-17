# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
    "sanitize_language_model",
]


def sanitize_language_model(language_model, weights, prefix):
    """Sanitize the language model's weights, named from prefix: "model." in the
    text model, "model.language_model." in a multimodal one."""
    args = language_model.config
    sanitized = {}
    first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers
    for k, v in weights.items():
        if any(
            s in k
            for s in (
                "self_attn.rotary_emb",
                "input_max",
                "input_min",
                "output_max",
                "output_min",
            )
        ):
            continue

        # KV-shared layers reuse K/V from earlier layers — drop their projections
        if any(
            s in k
            for s in (".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.k_norm")
        ):
            try:
                layer_idx = int(k.split("layers.")[1].split(".")[0])
                if layer_idx >= first_kv_shared:
                    continue
            except (IndexError, ValueError):
                pass

        if k.endswith(".experts.gate_up_proj"):
            base = k.removesuffix(".gate_up_proj")
            gate, up = map(mx.contiguous, mx.split(v, 2, axis=-2))
            sanitized[f"{base}.switch_glu.gate_proj.weight"] = gate
            sanitized[f"{base}.switch_glu.up_proj.weight"] = up
            continue

        if k.endswith(".experts.down_proj"):
            base = k.removesuffix(".down_proj")
            sanitized[f"{base}.switch_glu.down_proj.weight"] = v
            continue

        sanitized[k] = v

    return sanitized


def sanitize(model, weights):
    return sanitize_language_model(model.model, weights, prefix="model.")
