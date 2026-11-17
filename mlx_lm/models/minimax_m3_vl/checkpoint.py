# Copyright © 2025 Apple Inc.

from typing import Dict

import mlx.core as mx

from mlx_lm.layers.multimodal import from_llava_layout

__all__ = [
    "sanitize",
]


def sanitize(model, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    args = model.args.text
    # The checkpoints, and older MLX conversions, have the LLaVA layout. The
    # vision tower and the projectors are not used.
    skip = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
    out = {
        from_llava_layout(k): v
        for k, v in weights.items()
        if not k.removeprefix("model.").startswith(skip)
    }
    if args.tie_word_embeddings:
        out.pop("lm_head.weight", None)

    # MXFP8 stores e4m3 bytes plus one E8M0 exponent per 32 values, which is
    # MLX's mxfp8 layout once the bytes are read as uint32.
    for k in [k for k in out if k.endswith(".weight_scale_inv")]:
        base = k[: -len(".weight_scale_inv")]
        out[f"{base}.scales"] = out.pop(k)
        out[f"{base}.weight"] = out[f"{base}.weight"].view(mx.uint32)

    # Compressed-tensors NVFP4 experts store w = fp4 * scale / G. MLX
    # dequantizes with global_scale / (6 * 448), so its global scale is 2688 / G.
    for k in [k for k in out if k.endswith(".weight_packed")]:
        base = k[: -len(".weight_packed")]
        out[f"{base}.weight"] = out.pop(k).view(mx.uint32)
        out[f"{base}.scales"] = out.pop(f"{base}.weight_scale")
        out[f"{base}.global_scale"] = 2688.0 / out.pop(f"{base}.weight_global_scale")
        out.pop(f"{base}.input_global_scale", None)

    experts = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))
    for i in range(args.num_hidden_layers):
        moe = f"model.language_model.layers.{i}.block_sparse_moe"
        # Some exports keep a dense layer's MLP under block_sparse_moe. Its
        # gate is named gate_proj, the router's is gate, so this is exact.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if f"{moe}.{proj}.weight" not in out:
                continue
            mlp = f"model.language_model.layers.{i}.mlp"
            for part in ("weight", "scales", "biases"):
                v = out.pop(f"{moe}.{proj}.{part}", None)
                if v is not None:
                    out[f"{mlp}.{proj}.{part}"] = v

        # Some conversions fold the shared expert in as one more expert.
        # The expert axis is not the packed one, so the split is exact
        # on quantized weights.
        n = args.num_local_experts
        for proj in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            w = out.get(f"{moe}.switch_mlp.{proj}.weight")
            if w is None or w.shape[0] != n + 1:
                continue
            for part in ("weight", "scales", "biases"):
                v = out.pop(f"{moe}.switch_mlp.{proj}.{part}", None)
                if v is None:
                    continue
                out[f"{moe}.switch_mlp.{proj}.{part}"] = mx.contiguous(v[:n])
                out[f"{moe}.shared_experts.{proj}.{part}"] = mx.contiguous(v[n])

        for owner in (f"{moe}.switch_mlp", f"{moe}.shared_experts"):
            for part in ("weight", "scales", "biases"):
                fused = out.pop(f"{owner}.gate_up_proj.{part}", None)
                if fused is None:
                    continue
                gate, up = mx.split(fused, 2, axis=-2)
                out[f"{owner}.gate_proj.{part}"] = mx.contiguous(gate)
                out[f"{owner}.up_proj.{part}"] = mx.contiguous(up)

        if f"{moe}.experts.0.w1.weight" not in out:
            continue
        for src, dst in experts:
            for part in ("weight", "scales", "biases", "global_scale"):
                if f"{moe}.experts.0.{src}.{part}" not in out:
                    continue
                out[f"{moe}.switch_mlp.{dst}.{part}"] = mx.stack(
                    [
                        out.pop(f"{moe}.experts.{e}.{src}.{part}")
                        for e in range(args.num_local_experts)
                    ]
                )

    return out
