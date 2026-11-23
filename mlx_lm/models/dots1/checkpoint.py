# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        if l >= model.args.first_k_dense_replace:
            for _, m in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(model.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.experts.{m}.{k}"] = mx.stack(to_join)

    return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
