# Copyright © 2025 Apple Inc.

from typing import Dict

import mlx.core as mx

from mlx_lm.models.kimi_linear.layers import KimiDeltaAttention, KimiSparseMoE

__all__ = [
    "sanitize",
]


def sanitize(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for layer_idx, layer in enumerate(model.layers):
        prefix = f"model.layers.{layer_idx}"

        if isinstance(layer.mlp, KimiSparseMoE):
            src_prefix = f"{prefix}.block_sparse_moe"
            dst_prefix = f"{prefix}.mlp"
            for src, dst in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]:
                key = f"{src_prefix}.experts.0.{src}.weight"
                if key in weights:
                    stacked = [
                        weights.pop(f"{src_prefix}.experts.{i}.{src}.weight")
                        for i in range(model.args.num_experts)
                    ]
                    weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(stacked)

            for name in ("gate_proj", "up_proj", "down_proj"):
                src_key = f"{src_prefix}.shared_experts.{name}.weight"
                if src_key in weights:
                    weights[f"{dst_prefix}.shared_experts.{name}.weight"] = weights.pop(
                        src_key
                    )

            gate_key = f"{src_prefix}.gate.weight"
            if gate_key in weights:
                weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

            bias_key = f"{src_prefix}.gate.e_score_correction_bias"
            if bias_key in weights:
                weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(bias_key)

        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, KimiDeltaAttention):
            attn_prefix = f"{prefix}.self_attn"
            for src_name, dst_name in (
                ("q_conv1d", "q_conv"),
                ("k_conv1d", "k_conv"),
                ("v_conv1d", "v_conv"),
            ):
                src_key = f"{attn_prefix}.{src_name}.weight"
                if src_key in weights:
                    w = weights.pop(src_key)
                    if w.ndim == 3:
                        w = w.moveaxis(2, 1)
                    weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
            dt_key = f"{attn_prefix}.dt_bias"
            if dt_key in weights:
                if weights[dt_key].ndim > 1:
                    weights[dt_key] = mx.reshape(weights[dt_key], (-1,))

        attn_prefix = f"{prefix}.self_attn"
        kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
        if kv_b_key in weights:
            qk_nope = model.args.qk_nope_head_dim or model.args.head_dim
            v_head = model.args.v_head_dim or model.args.head_dim
            head_dim = qk_nope + v_head
            num_heads = model.args.num_attention_heads

            quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
            v = weights.pop(kv_b_key)

            if quantized:
                dims = model.args.kv_lora_rank
                scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                bits = (v.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

            v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, qk_nope:, :])

            if quantized:
                wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

            weights[f"{attn_prefix}.embed_q.weight"] = wk
            weights[f"{attn_prefix}.unembed_out.weight"] = wv

    return weights
