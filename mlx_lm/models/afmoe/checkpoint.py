# Copyright © 2024 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Remove unused precomputed rotary freqs
    weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    # Stack experts weights for SwitchGLU
    for l in range(model.args.num_hidden_layers):
        if l < model.args.num_dense_layers:
            continue
        prefix = f"model.layers.{l}"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                        for e in range(model.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.experts.{n}.{k}"] = mx.stack(to_join)

    return weights
