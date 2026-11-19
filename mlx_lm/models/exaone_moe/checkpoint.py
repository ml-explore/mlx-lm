# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    new_weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
    weights = new_weights

    for l in range(model.args.num_hidden_layers):
        if not model.args.is_moe_layer[l]:
            continue

        prefix = f"model.layers.{l}"

        bias_key = f"{prefix}.mlp.e_score_correction_bias"
        if bias_key in weights:
            weights[f"{prefix}.mlp.gate.e_score_correction_bias"] = weights.pop(
                bias_key
            )

        for m in ["gate_proj", "down_proj", "up_proj"]:
            for k in ["weight", "scales", "biases"]:
                first_key = f"{prefix}.mlp.experts.0.{m}.{k}"
                last_key = f"{prefix}.mlp.experts.{model.args.num_experts - 1}.{m}.{k}"
                if first_key in weights and last_key in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                        for e in range(model.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    return weights
