# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    def is_mpt_layer(key):
        subkeys = key.split(".")
        if len(subkeys) < 3:
            return False
        if subkeys[1] == "layers" and int(subkeys[2]) >= model.args.num_hidden_layers:
            return True
        return False

    new_weights = {}
    for k, v in weights.items():
        if is_mpt_layer(k):
            continue
        else:
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
        prefix = f"model.layers.{l}.self_attn"
        if f"{prefix}.kv_b_proj.weight" in weights:
            quantized = f"{prefix}.kv_b_proj.scales" in weights
            v = weights.pop(f"{prefix}.kv_b_proj.weight")
            head_dim = model.args.qk_nope_head_dim + model.args.v_head_dim

            if quantized:
                dims = model.args.kv_lora_rank
                scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                # Try to infer bits and group size
                bits = (v.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)
            num_heads = model.args.num_attention_heads
            v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, : model.args.qk_nope_head_dim, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, model.args.qk_nope_head_dim :, :])
            if quantized:
                wk, wk_scales, wk_biases = mx.quantize(
                    wk, bits=bits, group_size=group_size
                )
                wv, wv_scales, wv_biases = mx.quantize(
                    wv, bits=bits, group_size=group_size
                )
                weights[f"{prefix}.embed_q.scales"] = wk_scales
                weights[f"{prefix}.unembed_out.scales"] = wv_scales
                weights[f"{prefix}.embed_q.biases"] = wk_biases
                weights[f"{prefix}.unembed_out.biases"] = wv_biases
            weights[f"{prefix}.embed_q.weight"] = wk
            weights[f"{prefix}.unembed_out.weight"] = wv

    return weights
