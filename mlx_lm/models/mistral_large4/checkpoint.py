# Copyright © 2026 Apple Inc.

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.layers.moe import SwitchLinear
from mlx_lm.models.mistral_large4.layers import Fp8Linear, Fp8SwitchLinear

__all__ = [
    "sanitize",
]

_RENAMES = [
    (r"^tok_embeddings\.", "model.embed_tokens."),
    (r"^norm\.", "model.norm."),
    (r"^output\.", "lm_head."),
    (r"^lerope\.alpha$", "model.rope.alpha"),
    (r"^layers\.", "model.layers."),
    (r"\.attention_norm\.", ".input_layernorm."),
    (r"\.ffn_norm\.", ".post_attention_layernorm."),
    (r"\.ssmax_scaling\.s$", ".softmax_scaling.scale"),
    (r"\.ssmax_scaling\.b$", ".softmax_scaling.bias"),
    (r"\.attention\.indexer\.", ".self_attn.indexer."),
    (r"\.attention\.softmax_scaling\.", ".self_attn.softmax_scaling."),
    (r"\.attention\.wq\.", ".self_attn.q_proj."),
    (r"\.attention\.wkv_a_with_mqa\.", ".self_attn.kv_a_proj_with_mqa."),
    (r"\.attention\.kv_a_norm\.", ".self_attn.kv_a_layernorm."),
    (r"\.attention\.wkv_b\.", ".self_attn.kv_b_proj."),
    (r"\.attention\.wo\.", ".self_attn.o_proj."),
    (r"\.feed_forward\.", ".mlp."),
    (r"\.shared_experts\.", ".mlp.shared_experts."),
    (r"\.experts\.", ".mlp.experts."),
    (r"\.gate\.weight$", ".mlp.gate.weight"),
    (r"\.w1\.", ".gate_proj."),
    (r"\.w2\.", ".down_proj."),
    (r"\.w3\.", ".up_proj."),
    (r"\.qscale_weight$", ".weight_scale_inv"),
]

_DROP_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
    "mtp_dspark.",
)


def _fp8_as_mxfp8(weight, scale, block=128):
    # Power-of-two block scales are exact E8M0 scales, one per row and 32 columns.
    (m, n), (sm, sn) = weight.shape, scale.shape
    if (sm, sn) != (-(-m // block), -(-n // block)) or n % 32:
        raise ValueError(f"FP8 blocks of {weight.shape} / {scale.shape} are not mxfp8.")
    e8m0 = (mx.log2(scale.astype(mx.float32)) + 127).astype(mx.uint8)
    # The last block in each dimension can be smaller than the block size.
    e8m0 = mx.repeat(e8m0, block, axis=0)[:m]
    e8m0 = mx.repeat(e8m0, block // 32, axis=1)[:, : n // 32]
    return weight.view(mx.uint32), e8m0


def sanitize(model, weights):
    args = model.args.text

    new_weights = {}
    for k, v in weights.items():
        # Converted weights already have MLX names.
        if not k.startswith(("model.", "lm_head.")):
            if k.startswith(_DROP_PREFIXES):
                continue
            layer = re.match(r"layers\.(\d+)\.", k)
            if layer and int(layer.group(1)) >= args.num_hidden_layers:
                continue
            for pattern, repl in _RENAMES:
                k = re.sub(pattern, repl, k)
        new_weights[k] = v
    weights = new_weights

    # The FP8 layers stay FP8, as mxfp8. The quantization_config tells the
    # FP8 checkpoint, and a convert without -q, from other mxfp8 builds.
    fp8 = (model.args.quantization_config or {}).get("quant_method") == "fp8"
    scale_keys = [k for k in weights if k.endswith(".weight_scale_inv")]
    if scale_keys and not fp8:
        raise ValueError("FP8 weights need quant_method fp8 in quantization_config.")
    exact = []
    for k in scale_keys:
        wk = k.removesuffix("_scale_inv")
        scale = weights.pop(k)
        exact.append(mx.all(mx.power(2.0, mx.round(mx.log2(scale))) == scale))
        weights[wk], weights[f"{wk.removesuffix('weight')}scales"] = _fp8_as_mxfp8(
            weights[wk], scale
        )
    if exact and not mx.all(mx.stack(exact)).item():
        raise ValueError("FP8 block scales must be powers of two to run as mxfp8.")

    for l in range(args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        for m in ["gate_proj", "down_proj", "up_proj"]:
            for k in ["weight", "scales"]:
                keys = [
                    f"{prefix}.mlp.experts.{e}.{m}.{k}"
                    for e in range(args.n_routed_experts)
                ]
                # A pipeline stage can get some experts of a layer that it
                # does not run, from a file shared with its own layers.
                if all(key in weights for key in keys):
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                        [weights.pop(key) for key in keys]
                    )

        attn = f"{prefix}.self_attn"
        if f"{attn}.kv_b_proj.weight" in weights:
            nope = args.qk_nope_head_dim
            if not args.no_rope_layers[l]:
                nope += args.qk_rope_head_dim
            w = weights.pop(f"{attn}.kv_b_proj.weight")
            w = w.reshape(args.num_attention_heads, nope + args.v_head_dim, -1)
            weights[f"{attn}.embed_q.weight"] = mx.contiguous(
                w[:, :nope, :].swapaxes(-1, -2)
            )
            weights[f"{attn}.unembed_out.weight"] = mx.contiguous(w[:, nope:, :])

    if fp8:
        layers = []
        for p, m in model.named_modules():
            if f"{p}.scales" not in weights:
                continue
            if isinstance(m, SwitchLinear):
                layer = Fp8SwitchLinear(m.input_dims, m.output_dims, m.num_experts)
                layers.append((p, layer))
            elif isinstance(m, nn.Linear):
                out_dims, in_dims = m.weight.shape
                layers.append((p, Fp8Linear(in_dims, out_dims)))
        model.update_modules(tree_unflatten(layers))

    return weights
