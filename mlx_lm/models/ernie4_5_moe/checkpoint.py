# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    remove_patterns = [
        "mtp_block.",
        "mtp_linear_proj.",
        "mtp_hidden_norm.",
        "mtp_emb_norm.",
    ]

    weights = {
        key: value
        for key, value in weights.items()
        if not any(pattern in key for pattern in remove_patterns)
    }

    # Stack experts
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        # Only MoE layers have a gate, and weights may hold only some layers.
        if f"{prefix}.mlp.gate.weight" in weights:
            old = f"{prefix}.mlp.moe_statics.e_score_correction_bias"
            new = f"{prefix}.mlp.e_score_correction_bias"
            bias = weights.pop(old, weights.get(new, None))
            if bias is None:
                bias = mx.zeros((model.args.moe_num_experts,))
            # The checkpoint stores [1, E]. Converted weights already have [E].
            weights[new] = bias.squeeze(0) if bias.ndim == 2 else bias
        for m in ["gate_proj", "down_proj", "up_proj"]:
            if f"{prefix}.mlp.experts.0.{m}.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.mlp.experts.{e}.{m}.weight")
                    for e in range(model.args.moe_num_experts)
                ]
                weights[f"{prefix}.mlp.switch_mlp.{m}.weight"] = mx.stack(to_join)

    return weights
