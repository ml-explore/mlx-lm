# Copyright © 2026 Apple Inc.

import re

import mlx.core as mx

__all__ = [
    "sanitize",
]


_DROP_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")
_EXPERT_KEY = r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)"
_EXPERT_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _rename(key: str) -> str:
    """Map one DeepSeek checkpoint name to its MLX name."""
    if key == "head.weight":
        return "lm_head.weight"
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    key = f"model.{key}"
    key = key.replace(".attn.", ".self_attn.")
    key = key.replace(".ffn.", ".mlp.")
    key = key.replace(".attn_norm.", ".input_layernorm.")
    key = key.replace(".ffn_norm.", ".post_attention_layernorm.")
    key = key.replace(".hc_attn_", ".attn_hc.")
    key = key.replace(".hc_ffn_", ".ffn_hc.")
    key = key.replace(".attn_sink", ".sinks")
    key = key.replace(".wq_a.", ".q_a_proj.")
    key = key.replace(".wq_b.", ".q_b_proj.")
    key = key.replace(".wkv.", ".kv_proj.")
    key = key.replace(".wgate.", ".gate_proj.")
    key = key.replace(".wo_a.", ".o_a_proj.")
    key = key.replace(".wo_b.", ".o_b_proj.")
    key = key.replace(".q_norm.", ".q_a_norm.")
    key = key.replace(".compressor.norm.", ".compressor.kv_norm.")
    key = key.replace(".indexer.wk.", ".indexer.k_proj.")
    key = key.replace(".gate.bias", ".gate.e_score_correction_bias")
    key = key.replace(".shared_experts.w1.", ".shared_experts.gate_proj.")
    key = key.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
    key = key.replace(".shared_experts.w3.", ".shared_experts.up_proj.")
    return key


def _repack(weight: mx.array, scale: mx.array) -> tuple[mx.array, mx.array]:
    """Put an fp8 or fp4 weight and its ue8m0 scale in MLX's quantized layout."""
    if scale.dtype != mx.uint8:
        scale = (mx.log2(scale.astype(mx.float32)) + 127).astype(mx.uint8)
    if scale.shape[-2] != weight.shape[-2]:
        # fp8 keeps one scale per 32x32 tile, MLX one per output row.
        scale = mx.repeat(scale, 32, axis=-2)[..., : weight.shape[-2], :]
    return weight.view(mx.uint32), scale


def sanitize(model, weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Rename the checkpoint keys and keep the fp8/fp4 weights packed."""
    # A converted checkpoint already has MLX names; the published one never
    # starts a key with "model.".
    if any(key.startswith("model.") for key in weights):
        return weights
    text = {
        key: value
        for key, value in weights.items()
        if not key.startswith(_DROP_PREFIXES) and ".engram." not in key
    }
    clean = {}
    experts = {}
    for key, value in text.items():
        expert = re.fullmatch(_EXPERT_KEY, key)
        if expert is not None:
            layer, index, proj, kind = expert.groups()
            layer, index = int(layer), int(index)
            rows = experts.setdefault((layer, proj), {})
            rows.setdefault(index, {})[kind] = value
            continue
        # bias_vl routes image tokens, which the text model never sees.
        if key.endswith(".gate.bias_vl"):
            continue
        if key.endswith(".scale") and f"{key.removesuffix('.scale')}.weight" in text:
            continue
        scale = None
        if key.endswith(".weight"):
            scale = text.get(f"{key.removesuffix('.weight')}.scale")
        name = _rename(key)
        if scale is not None:
            value, scale = _repack(value, scale)
        if name.endswith(".o_a_proj.weight"):
            # wo_a is block diagonal, so split it back into its groups.
            value = value.reshape(model.args.o_groups, -1, value.shape[-1])
            if scale is not None:
                scale = scale.reshape(model.args.o_groups, -1, scale.shape[-1])
        clean[name] = value
        if scale is not None:
            clean[f"{name.removesuffix('weight')}scales"] = scale

    n_experts = model.args.n_routed_experts
    for (layer, proj), rows in experts.items():
        if len(rows) != n_experts:
            raise ValueError(
                f"Layer {layer} has {len(rows)} {proj} expert weights, "
                f"but the config declares {n_experts} experts."
            )
        parts = [rows[index] for index in sorted(rows)]
        name = f"model.layers.{layer}.mlp.experts.switch_mlp.{_EXPERT_PROJ[proj]}"
        value = mx.stack([part["weight"] for part in parts])
        if "scale" in parts[0]:
            scales = mx.stack([part["scale"] for part in parts])
            value, scales = _repack(value, scales)
            clean[f"{name}.scales"] = scales
        clean[f"{name}.weight"] = value
    return clean
