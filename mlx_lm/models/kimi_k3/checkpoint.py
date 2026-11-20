# Copyright © 2026 Apple Inc.

import re
from typing import Dict

import mlx.core as mx

from mlx_lm.models.checkpoint import from_text_layout
from mlx_lm.models.kimi_k3.layers import KimiK3DeltaAttention, KimiK3SparseMoE

__all__ = [
    "sanitize",
    "sanitize_language_model",
]

# Only the language model runs here.
_ENCODERS = ("vision_tower", "vision_model", "multi_modal_projector", "mm_projector")


def sanitize_language_model(
    language_model, weights: Dict[str, mx.array], prefix: str
) -> Dict[str, mx.array]:
    """Sanitize the language model's weights, whose names start with prefix."""
    args = language_model.args
    weights = {
        k: v for k, v in weights.items() if not k.startswith((f"{prefix}mtp", "mtp"))
    }
    layer_re = re.compile(re.escape(prefix) + r"layers\.(\d+)\.")
    weights = {
        k: v
        for k, v in weights.items()
        if not (m := layer_re.match(k)) or int(m.group(1)) < args.num_hidden_layers
    }

    res_renames = []
    for src in ("self_attention_res", "mlp_res", "output_attn_res"):
        res_renames.append((f"{src}.proj_weight", f"{src}_proj.weight"))
        res_renames.append((f"{src}.norm_weight", f"{src}_norm.weight"))
    for k in list(weights):
        for pat, dst in res_renames:
            if k.endswith(pat):
                weights[k[: -len(pat)] + dst] = weights.pop(k)
                break

    for layer_idx, layer in enumerate(language_model.layers):
        lp = f"{prefix}layers.{layer_idx}"

        if isinstance(layer.mlp, KimiK3SparseMoE):
            src_prefix = f"{lp}.block_sparse_moe"
            dst_prefix = f"{lp}.mlp"
            for src, dst in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]:
                if f"{src_prefix}.experts.0.{src}.weight_packed" in weights:
                    packed = mx.stack(
                        [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight_packed")
                            for i in range(args.num_experts)
                        ]
                    )
                    scales = mx.stack(
                        [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight_scale")
                            for i in range(args.num_experts)
                        ]
                    )
                    weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = packed.view(
                        mx.uint32
                    )
                    weights[f"{dst_prefix}.switch_mlp.{dst}.scales"] = scales
                else:
                    for suffix in ("weight", "scales", "biases"):
                        if f"{src_prefix}.experts.0.{src}.{suffix}" in weights:
                            weights[f"{dst_prefix}.switch_mlp.{dst}.{suffix}"] = (
                                mx.stack(
                                    [
                                        weights.pop(
                                            f"{src_prefix}.experts.{i}.{src}.{suffix}"
                                        )
                                        for i in range(args.num_experts)
                                    ]
                                )
                            )

            for name in (
                "shared_experts.gate_proj",
                "shared_experts.up_proj",
                "shared_experts.down_proj",
                "routed_expert_down_proj",
                "routed_expert_up_proj",
                "routed_expert_norm",
                "gate",
                "switch_mlp.gate_proj",
                "switch_mlp.up_proj",
                "switch_mlp.down_proj",
            ):
                for suffix in ("weight", "scales", "biases"):
                    src_key = f"{src_prefix}.{name}.{suffix}"
                    if src_key in weights:
                        weights[f"{dst_prefix}.{name}.{suffix}"] = weights.pop(src_key)

            for bias_key in (
                f"{src_prefix}.gate.e_score_correction_bias",
                f"{src_prefix}.e_score_correction_bias",
            ):
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                        bias_key
                    )

        attn = getattr(layer, "self_attn", None)
        ap = f"{lp}.self_attn"
        if isinstance(attn, KimiK3DeltaAttention):
            for src_name, dst_name in (
                ("q_conv1d", "q_conv"),
                ("k_conv1d", "k_conv"),
                ("v_conv1d", "v_conv"),
            ):
                src_key = f"{ap}.{src_name}.weight"
                if src_key in weights:
                    w = weights.pop(src_key)
                    if w.ndim == 3:
                        w = w.moveaxis(2, 1)
                    weights[f"{ap}.{dst_name}.conv.weight"] = w
            for name in ("dt_bias", "A_log"):
                key = f"{ap}.{name}"
                if key in weights and weights[key].ndim > 1:
                    weights[key] = mx.reshape(weights[key], (-1,))
            a_log_key = f"{ap}.A_log"
            num_heads = args.linear_attn_config["num_heads"]
            if a_log_key in weights and weights[a_log_key].shape[0] > num_heads:
                weights[a_log_key] = weights[a_log_key][:num_heads]

            if f"{ap}.qkv_proj.weight" not in weights:
                for suffix in ("weight", "scales", "biases"):
                    parts = [f"{ap}.{p}_proj.{suffix}" for p in "qkv"]
                    if all(p in weights for p in parts):
                        weights[f"{ap}.qkv_proj.{suffix}"] = mx.concatenate(
                            [weights.pop(p) for p in parts], axis=0
                        )
            conv_parts = [f"{ap}.{p}_conv.conv.weight" for p in "qkv"]
            if f"{ap}.qkv_conv.conv.weight" not in weights and all(
                p in weights for p in conv_parts
            ):
                weights[f"{ap}.qkv_conv.conv.weight"] = mx.concatenate(
                    [weights.pop(p) for p in conv_parts], axis=0
                )

        kv_b_key = f"{ap}.kv_b_proj.weight"
        if kv_b_key in weights:
            qk_nope = args.qk_nope_head_dim
            v_head = args.v_head_dim
            head_dim = qk_nope + v_head
            num_heads = args.num_attention_heads

            quantized = f"{ap}.kv_b_proj.scales" in weights
            v = weights.pop(kv_b_key)

            if quantized:
                dims = args.kv_lora_rank
                scales = weights.pop(f"{ap}.kv_b_proj.scales")
                biases = weights.pop(f"{ap}.kv_b_proj.biases")
                bits = (v.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

            v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, qk_nope:, :])

            if quantized:
                wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                weights[f"{ap}.embed_q.scales"] = wk_s
                weights[f"{ap}.embed_q.biases"] = wk_b
                weights[f"{ap}.unembed_out.scales"] = wv_s
                weights[f"{ap}.unembed_out.biases"] = wv_b

            weights[f"{ap}.embed_q.weight"] = wk
            weights[f"{ap}.unembed_out.weight"] = wv

    return weights


def sanitize(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    # The checkpoints have the text model's layout, under language_model in the
    # multimodal ones, as do older MLX conversions.
    weights = {
        from_text_layout(k.removeprefix("language_model.")): v
        for k, v in weights.items()
        if not k.removeprefix("model.").startswith(_ENCODERS)
    }
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return sanitize_language_model(
        model.model.language_model, weights, prefix="model.language_model."
    )
