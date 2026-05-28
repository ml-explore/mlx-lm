# Copyright © 2026 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn


class MultiLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )

    def __call__(self, x, transpose=True):
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight

    def to_quantized(
        self,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ):
        num_heads, output_dims, input_dims = self.weight.shape
        ql = QuantizedMultiLinear(
            input_dims, output_dims, num_heads, group_size, bits, mode
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
        return ql


class QuantizedMultiLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_heads: int,
        group_size: int,
        bits: int,
        mode: str,
    ):
        super().__init__()

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        self.freeze()

    def __call__(self, x, transpose=True):
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


def split_kv_b_proj_weights(
    weights,
    prefix: str,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
):
    kv_b_proj = f"{prefix}.kv_b_proj"
    if f"{kv_b_proj}.weight" not in weights:
        return weights

    quantized = f"{kv_b_proj}.scales" in weights
    weight = weights.pop(f"{kv_b_proj}.weight")
    head_dim = qk_nope_head_dim + v_head_dim

    if quantized:
        scales = weights.pop(f"{kv_b_proj}.scales")
        biases = weights.pop(f"{kv_b_proj}.biases", None)
        # Infer bits and group size from the packed shapes.
        bits = (weight.shape[-1] * 32) // kv_lora_rank
        group_size = kv_lora_rank // scales.shape[-1]
        if biases is None:
            biases = mx.zeros_like(scales)
        weight = mx.dequantize(weight, scales, biases, bits=bits, group_size=group_size)

    weight = weight.reshape(num_heads, head_dim, -1)
    wk = mx.contiguous(weight[:, :qk_nope_head_dim, :].swapaxes(-1, -2))
    wv = mx.contiguous(weight[:, qk_nope_head_dim:, :])

    if quantized:
        # Re-quantize the split projections so they reload as QuantizedMultiLinear.
        wk, wk_scales, wk_biases = mx.quantize(wk, bits=bits, group_size=group_size)
        wv, wv_scales, wv_biases = mx.quantize(wv, bits=bits, group_size=group_size)
        weights[f"{prefix}.embed_q.scales"] = wk_scales
        weights[f"{prefix}.embed_q.biases"] = wk_biases
        weights[f"{prefix}.unembed_out.scales"] = wv_scales
        weights[f"{prefix}.unembed_out.biases"] = wv_biases

    weights[f"{prefix}.embed_q.weight"] = wk
    weights[f"{prefix}.unembed_out.weight"] = wv
    return weights


def shard_mla_projections(attn, group, num_heads_attr: str):
    num_heads = getattr(attn, num_heads_attr)
    group_size = group.size()
    if num_heads % group_size != 0:
        raise ValueError(
            f"Cannot shard {num_heads_attr}={num_heads} across " f"{group_size} ranks."
        )

    local_heads = num_heads // group_size
    sh = group.rank() * local_heads
    eh = sh + local_heads

    def shard_heads(w):
        return w[sh:eh]

    setattr(attn, num_heads_attr, local_heads)
    attn.embed_q.apply(shard_heads)
    attn.unembed_out.apply(shard_heads)
