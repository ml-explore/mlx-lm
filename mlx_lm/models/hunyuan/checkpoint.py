# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):

    if "model.layers.0.mlp.gate_and_up_proj.weight" in weights:
        new_weights = {}
        D = model.args.hidden_size
        n_kv_heads = model.args.num_key_value_heads
        n_kv_groups = model.args.num_attention_heads // n_kv_heads
        head_dim = D // model.args.num_attention_heads
        for k, v in weights.items():
            if "qkv_proj" in k:
                v = v.reshape(n_kv_heads, n_kv_groups + 2, head_dim, -1)
                splits = v.split([n_kv_groups, n_kv_groups + 1], axis=1)
                for k_up, v_new in zip(["q_proj", "k_proj", "v_proj"], splits):
                    k_new = k.replace("qkv_proj", k_up)
                    new_weights[k_new] = mx.flatten(v_new, 0, 2)
            elif "gate_and_up_proj" in k:
                splits = v.split(2, axis=0)
                for k_up, v_new in zip(["up_proj", "gate_proj"], splits):
                    k_new = k.replace("gate_and_up_proj", k_up)
                    new_weights[k_new] = v_new
            else:
                new_weights[k] = v
        weights = new_weights

    if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
        return weights
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                        for e in range(model.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(to_join)
    return weights
