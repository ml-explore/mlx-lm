# Copyright © 2024 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    def keep(key):
        return (
            "vision_tower" not in key
            and "rotary_emb" not in key
            and "multi_modal_projector" not in key
        )

    weights = {k: v for k, v in weights.items() if keep(k)}
    # Stack experts
    for l in range(model.args.text_config.num_hidden_layers):
        prefix = f"language_model.model.layers.{l}"
        for m in [("gate_proj"), ("down_proj"), ("up_proj")]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                        for e in range(model.args.text_config.n_routed_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

    return weights
