# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    """Dequantize FP8 weights and restructure MoE experts."""

    def dequant(weight, scale_inv):
        dtype = mx.bfloat16
        weight = mx.from_fp8(weight, dtype=mx.bfloat16)
        bs = 128  # block size
        m, n = weight.shape
        pad_bottom = (-m) % bs
        pad_side = (-n) % bs
        weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
        weight = weight.reshape(((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs))
        weight = (weight * scale_inv[:, None, :, None]).reshape(
            m + pad_bottom, n + pad_side
        )
        return weight[:m, :n].astype(dtype)

    # Dequantize
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

    # Step 2: Handle MoE expert weights restructuring
    if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
        return weights

    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for orig_name, new_name in mapping.items():
            if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                to_join = [
                    weights.pop(
                        f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                    )
                    for e in range(model.args.num_local_experts)
                ]
                weights[f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"] = (
                    mx.stack(to_join)
                )

    return weights
