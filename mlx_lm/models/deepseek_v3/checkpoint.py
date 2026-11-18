# Copyright © 2024 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
    "sanitize_language_model",
]


def sanitize_language_model(language_model, weights, prefix):
    """Sanitize the language model's weights, whose names start with prefix."""
    args = language_model.args

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

    # Remap for int4
    new_weights = {}
    for k, v in weights.items():
        if k.endswith("weight_shape"):
            base = k.replace("weight_shape", "")
            new_weights[base + "weight"] = weights[base + "weight_packed"].view(
                mx.uint32
            )
            s = weights[base + "weight_scale"]
            new_weights[base + "scales"] = s
            new_weights[base + "biases"] = -8 * s
        elif not (k.endswith("weight_scale") or k.endswith("weight_packed")):
            new_weights[k] = v
    weights = new_weights

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
    for l in range(args.num_hidden_layers):
        layer = f"{prefix}layers.{l}"
        for _, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            for k in ["weight", "scales", "biases"]:
                if f"{layer}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{layer}.mlp.experts.{e}.{m}.{k}")
                        for e in range(args.n_routed_experts)
                    ]
                    weights[f"{layer}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        attn = f"{prefix}layers.{l}.self_attn"
        if f"{attn}.kv_b_proj.weight" in weights:
            quantized = f"{attn}.kv_b_proj.scales" in weights
            v = weights.pop(f"{attn}.kv_b_proj.weight")
            head_dim = args.qk_nope_head_dim + args.v_head_dim

            if quantized:
                dims = args.kv_lora_rank
                scales = weights.pop(f"{attn}.kv_b_proj.scales")
                biases = weights.pop(f"{attn}.kv_b_proj.biases")
                # Try to infer bits and group size
                bits = (v.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)
            num_heads = args.num_attention_heads
            v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, args.qk_nope_head_dim :, :])
            if quantized:
                wk, wk_scales, wk_biases = mx.quantize(
                    wk, bits=bits, group_size=group_size
                )
                wv, wv_scales, wv_biases = mx.quantize(
                    wv, bits=bits, group_size=group_size
                )
                weights[f"{attn}.embed_q.scales"] = wk_scales
                weights[f"{attn}.unembed_out.scales"] = wv_scales
                weights[f"{attn}.embed_q.biases"] = wk_biases
                weights[f"{attn}.unembed_out.biases"] = wv_biases
            weights[f"{attn}.embed_q.weight"] = wk
            weights[f"{attn}.unembed_out.weight"] = wv

    # Remove multi-token prediction layer and any unused precomputed rotary freqs
    return {
        k: v
        for k, v in weights.items()
        if not k.startswith(f"{prefix}layers.61") and "rotary_emb.inv_freq" not in k
    }


def sanitize(model, weights):
    return sanitize_language_model(model.model, weights, prefix="model.")
