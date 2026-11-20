# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.models.ministral3.layers import Fp8Linear, FusedGLU

__all__ = [
    "sanitize",
    "sanitize_language_model",
    "use_fp8_linears",
]


def use_fp8_linears(
    module: nn.Module, weights: dict[str, mx.array], prefix: str
) -> None:
    """Swap the linear layers that have an fp8 output scale for Fp8Linear."""
    fp8_modules = []
    for p, m in module.named_modules():
        if isinstance(m, nn.Linear) and f"{prefix}{p}.output_scale" in weights:
            out_dims, in_dims = m.weight.shape
            per_row = weights[f"{prefix}{p}.output_scale"].ndim == 1
            fp8_modules.append((p, Fp8Linear(in_dims, out_dims, per_row)))
    if fp8_modules:
        module.update_modules(tree_unflatten(fp8_modules))


def _fuse(weights, parts, fused):
    """Join the output rows of layers that read the same input.

    Returns False and changes nothing when the layers are stored differently,
    for example quantized with different bits.
    """
    keys = [{k[len(p) + 1 :] for k in weights if k.startswith(p + ".")} for p in parts]
    if "weight" not in keys[0] or any(k != keys[0] for k in keys):
        return False
    for k in keys[0]:
        vs = [weights[f"{p}.{k}"] for p in parts]
        if k == "output_scale" and vs[0].ndim > 0:
            return False
        if k not in ("output_scale", "bias") and any(
            v.dtype != vs[0].dtype or v.shape[-1] != vs[0].shape[-1] for v in vs
        ):
            return False

    rows = [weights[f"{p}.weight"].shape[-2] for p in parts]
    for k in keys[0]:
        vs = [weights.pop(f"{p}.{k}") for p in parts]
        if k == "output_scale":
            # Each fp8 layer has one scale. Fused, there is one per row.
            vs = [mx.full((n,), s, dtype=s.dtype) for n, s in zip(rows, vs)]
            weights[f"{fused}.{k}"] = mx.concatenate(vs)
        else:
            axis = -1 if k == "bias" else -2
            weights[f"{fused}.{k}"] = mx.concatenate(vs, axis=axis)
    return True


def _fp8_as_mxfp8(weights):
    """Read the e4m3 weights of fp8 layers as mxfp8 with unit (2^0) group scales."""
    for k in [k for k in weights if k.endswith(".output_scale")]:
        p = k.removesuffix(".output_scale")
        w = weights[f"{p}.weight"]
        if w.dtype == mx.uint8:
            weights[f"{p}.weight"] = w.view(mx.uint32)
            weights[f"{p}.scales"] = mx.full(
                (*w.shape[:-1], w.shape[-1] // 32), 127, mx.uint8
            )


def sanitize_language_model(language_model, weights, prefix):
    """Sanitize the language model's weights, whose names start with prefix."""
    # Remove unused precomputed rotary freqs
    weights = {
        k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
    }

    # fp8 layers keep the e4m3 bytes, and the scale moves to the layer.
    # Static activation scales have no consumer here.
    weights = {
        k.replace(".weight_scale_inv", ".output_scale"): v
        for k, v in weights.items()
        if "activation_scale" not in k
    }

    def fused(parts, name):
        if f"{name}.weight" in weights:
            return True
        # Only fp8 layers are fused. AWQ and the mixed-bit recipes look up
        # q_proj, v_proj and gate_proj by name.
        fp8 = f"{parts[0]}.output_scale" in weights
        return fp8 and _fuse(weights, parts, name)

    # Projections that read the same input run as one matmul.
    for l, layer in enumerate(language_model.layers):
        attn, mlp = f"{prefix}layers.{l}.self_attn", f"{prefix}layers.{l}.mlp"
        a = layer.self_attn
        if fused([f"{attn}.{n}_proj" for n in "qkv"], f"{attn}.qkv_proj"):
            rows = (a.n_heads + 2 * a.n_kv_heads) * a.head_dim
            a.qkv_proj = nn.Linear(language_model.args.hidden_size, rows, bias=False)
            del a.q_proj, a.k_proj, a.v_proj
        if fused([f"{mlp}.gate_proj", f"{mlp}.up_proj"], f"{mlp}.gate_up_proj"):
            out_dims, in_dims = layer.mlp.gate_proj.weight.shape
            layer.mlp = FusedGLU(in_dims, out_dims)

    _fp8_as_mxfp8(weights)
    use_fp8_linears(language_model, weights, prefix)
    return weights


def sanitize(model, weights):
    weights = sanitize_language_model(model.model, weights, prefix="model.")
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    # The output layer is outside the language model.
    use_fp8_linears(model, weights, prefix="")
    return weights
