# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    n_layers = model.args.num_hidden_layers
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for l in range(n_layers):
        prefix = f"model.layers.{l}"
        for m in ["gate_proj", "down_proj", "up_proj"]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                        for e in range(model.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        if f"{prefix}.mlp.gate.weight" in weights:
            weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(
                f"{prefix}.mlp.gate.weight"
            )

        prefix = f"{prefix}.attention"
        if f"{prefix}.kv_b_proj.weight" in weights:
            v = weights.pop(f"{prefix}.kv_b_proj.weight")
            nope = model.args.qk_nope_head_dim
            head_dim = nope + model.args.v_head_dim
            v = v.reshape(model.args.num_attention_heads, head_dim, -1)
            weights[f"{prefix}.embed_q.weight"] = mx.contiguous(
                v[:, :nope, :].swapaxes(-1, -2)
            )
            weights[f"{prefix}.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])

    # The checkpoint appends MTP layers after the base ones.
    return {
        k: v
        for k, v in weights.items()
        if not (
            k.startswith("model.layers.")
            and k.split(".")[2].isdigit()
            and int(k.split(".")[2]) >= n_layers
        )
    }
