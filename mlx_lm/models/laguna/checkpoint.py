# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Repacked checkpoints wrap every tensor in a `language_model.` prefix.
    if any(k.startswith("language_model.") for k in weights):
        prefix = "language_model."
        weights = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in weights.items()
        }

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}.mlp"

        # The original layout has a bare router and the bias under `experts.`.
        gate_weight = weights.pop(f"{prefix}.gate.weight", None)
        if gate_weight is not None:
            weights[f"{prefix}.gate.proj.weight"] = gate_weight
        bias = weights.pop(f"{prefix}.experts.e_score_correction_bias", None)
        if bias is not None:
            weights[f"{prefix}.gate.e_score_correction_bias"] = bias

        # It also stores experts individually; stack them for SwitchGLU.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                if f"{prefix}.experts.0.{proj}.{suffix}" not in weights:
                    continue
                weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{e}.{proj}.{suffix}")
                        for e in range(model.args.num_experts)
                    ]
                )
    return weights
