# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
    "sanitize_language_model",
]


def sanitize_language_model(language_model, weights, prefix):
    """Sanitize the language model's weights, whose names start with prefix."""
    args = language_model.args

    def broadcasts(scale_shape, weight_shape):
        if len(scale_shape) > len(weight_shape):
            return False
        pad = (1,) * (len(weight_shape) - len(scale_shape)) + tuple(scale_shape)
        return all(s in (1, w) for s, w in zip(pad, weight_shape))

    def dequant(weight, scale_inv):
        dtype = mx.bfloat16
        weight = mx.from_fp8(weight, dtype=dtype)
        # Per-tensor (rank 0) and per-expert ([E, 1, 1]) scales broadcast.
        if broadcasts(scale_inv.shape, weight.shape):
            return (weight * scale_inv).astype(dtype)
        bs = 128
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
        # Static activation scales have no consumer here.
        if k.endswith("activation_scale"):
            continue
        # Expert scales are named "experts.down_proj_scale_inv", so match
        # the suffix rather than a "weight_scale_inv" substring.
        if k.endswith("_scale_inv"):
            wk = k.replace("_scale_inv", "")
            new_weights[wk] = dequant(weights[wk], v)
        elif k not in new_weights:
            new_weights[k] = v
    weights = new_weights

    for l in range(args.num_hidden_layers):
        layer = f"{prefix}layers.{l}"

        # Handle fused gate_up_proj format (Mistral4NaiveMoe)
        gup_key = f"{layer}.mlp.experts.gate_up_proj"
        if gup_key in weights:
            gate_up = weights.pop(gup_key)
            gate, up = mx.split(gate_up, 2, axis=1)
            weights[f"{layer}.mlp.switch_mlp.gate_proj.weight"] = gate
            weights[f"{layer}.mlp.switch_mlp.up_proj.weight"] = up
        down_key = f"{layer}.mlp.experts.down_proj"
        if down_key in weights:
            weights[f"{layer}.mlp.switch_mlp.down_proj.weight"] = weights.pop(down_key)

        for m in ["gate_proj", "down_proj", "up_proj"]:
            for k in ["weight", "scales", "biases"]:
                if f"{layer}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{layer}.mlp.experts.{e}.{m}.{k}")
                        for e in range(args.n_routed_experts)
                    ]
                    weights[f"{layer}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        # Absorb kv_b_proj into embed_q / unembed_out.
        # TODO: affine only; non-affine modes have no biases key.
        attn = f"{layer}.self_attn"
        if f"{attn}.kv_b_proj.weight" in weights:
            quantized = f"{attn}.kv_b_proj.scales" in weights
            w = weights.pop(f"{attn}.kv_b_proj.weight")
            head_dim = args.qk_nope_head_dim + args.v_head_dim
            if quantized:
                dims = args.kv_lora_rank
                scales = weights.pop(f"{attn}.kv_b_proj.scales")
                biases = weights.pop(f"{attn}.kv_b_proj.biases")
                bits = (w.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                w = mx.dequantize(w, scales, biases, bits=bits, group_size=group_size)
            w = w.reshape(args.num_attention_heads, head_dim, -1)
            wk = mx.contiguous(w[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2))
            wv = mx.contiguous(w[:, args.qk_nope_head_dim :, :])
            if quantized:
                wk, wk_scales, wk_biases = mx.quantize(
                    wk, bits=bits, group_size=group_size
                )
                wv, wv_scales, wv_biases = mx.quantize(
                    wv, bits=bits, group_size=group_size
                )
                weights[f"{attn}.embed_q.scales"] = wk_scales
                weights[f"{attn}.embed_q.biases"] = wk_biases
                weights[f"{attn}.unembed_out.scales"] = wv_scales
                weights[f"{attn}.unembed_out.biases"] = wv_biases
            weights[f"{attn}.embed_q.weight"] = wk
            weights[f"{attn}.unembed_out.weight"] = wv

    return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}


def sanitize(model, weights):
    return sanitize_language_model(model.model, weights, prefix="model.")
