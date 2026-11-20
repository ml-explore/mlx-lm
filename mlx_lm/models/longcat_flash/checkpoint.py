# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for l in range(model.args.num_layers):
        prefix = f"model.layers.{l}"
        for _, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                        for e in range(model.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

    for l in range(model.args.num_layers):
        for i in range(2):
            prefix = f"model.layers.{l}.self_attn.{i}"
            kv_b_key = f"{prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                num_heads = model.args.num_attention_heads
                head_dim = model.args.qk_nope_head_dim + model.args.v_head_dim
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = model.args.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : model.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, model.args.qk_nope_head_dim :, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{prefix}.embed_q.scales"] = wk_s
                    weights[f"{prefix}.embed_q.biases"] = wk_b
                    weights[f"{prefix}.unembed_out.scales"] = wv_s
                    weights[f"{prefix}.unembed_out.biases"] = wv_b

                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

    new_weights = {}
    for k, v in weights.items():
        if k.startswith("model.mtp"):
            continue
        new_weights[k] = v
    return new_weights
