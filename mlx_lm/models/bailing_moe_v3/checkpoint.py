# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def _is_mtp_weight(key: str, num_hidden_layers: int) -> bool:
    prefix = "model.layers."
    if not key.startswith(prefix):
        return False
    layer_index = key[len(prefix) :].split(".", 1)[0]
    return layer_index.isdigit() and int(layer_index) >= num_hidden_layers


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    weights = {
        key: value
        for key, value in weights.items()
        if not _is_mtp_weight(key, model.args.num_hidden_layers)
    }
    for layer_index in range(model.args.num_hidden_layers):
        layer_prefix = f"model.layers.{layer_index}"
        if layer_index >= model.args.first_k_dense_replace:
            mlp_prefix = f"{layer_prefix}.mlp"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                expert_prefix = f"{mlp_prefix}.experts.0.{projection}"
                if f"{expert_prefix}.weight" not in weights:
                    continue
                expert_weights = [
                    weights.pop(f"{mlp_prefix}.experts.{expert}.{projection}.weight")
                    for expert in range(model.args.num_experts)
                ]
                weights[f"{mlp_prefix}.switch_mlp.{projection}.weight"] = mx.stack(
                    expert_weights
                )

        attention_prefix = f"{layer_prefix}.attention"
        # transformers 5 saves the KDA output projection as dense.
        dense_key = f"{attention_prefix}.dense.weight"
        if model.layers[layer_index].is_linear and dense_key in weights:
            weights[f"{attention_prefix}.o_proj.weight"] = weights.pop(dense_key)
        kv_b_key = f"{attention_prefix}.kv_b_proj.weight"
        if kv_b_key not in weights:
            continue
        num_heads = model.args.num_attention_heads
        head_dim = model.args.qk_nope_head_dim + model.args.v_head_dim
        if f"{attention_prefix}.kv_b_proj.scales" in weights:
            raise ValueError("Quantized Bailing V3 kv_b_proj weights are not supported")
        value = weights.pop(kv_b_key)
        value = value.reshape(num_heads, head_dim, -1)
        wk = mx.contiguous(value[:, : model.args.qk_nope_head_dim, :].swapaxes(-1, -2))
        wv = mx.contiguous(value[:, model.args.qk_nope_head_dim :, :])
        weights[f"{attention_prefix}.embed_q.weight"] = wk
        weights[f"{attention_prefix}.unembed_out.weight"] = wv

    for key, value in list(weights.items()):
        if "_conv1d.weight" in key and value.shape[-1] != 1:
            weights[key] = value.moveaxis(2, 1)
    return weights
