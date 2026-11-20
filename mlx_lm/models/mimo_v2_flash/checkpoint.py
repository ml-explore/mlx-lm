# Copyright © 2024 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    def dequant(weight, scale_inv):
        dtype = mx.bfloat16
        weight = mx.from_fp8(weight, dtype=mx.bfloat16)
        bs = 128  # block size
        m, n = weight.shape
        pad_bottom = bs * scale_inv.shape[0] - m
        pad_side = bs * scale_inv.shape[1] - n
        weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
        weight = weight.reshape(((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs))
        weight = (weight * scale_inv[:, None, :, None]).reshape(
            m + pad_bottom, n + pad_side
        )
        return weight[:m, :n].astype(dtype)

    # Dequantize fp8
    new_weights = {}
    for k, v in weights.items():
        if "weight_scale_inv" in k:
            scale_inv = v
            wk = k.replace("_scale_inv", "")
            weight = weights[wk]
            weight = dequant(weight, scale_inv)
            new_weights[wk] = weight
        elif k not in new_weights:
            new_weights[k] = v
    weights = new_weights

    # Stack experts
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        for _, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                        for e in range(model.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

    # Remove multi-token prediction layer
    return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}
