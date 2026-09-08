# Copyright © 2026 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def max_absorbed_queries(
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    cache_len: Optional[int] = None,
) -> int:
    """Query count below which folding into the query beats materializing K/V.

    Per head, with ``r = kv_lora_rank`` and ``d = qk_nope_head_dim +
    v_head_dim``, the absorbed path costs ``L*r*d + 2*L*T*r`` and materializing
    costs ``T*r*d + L*T*d``, where ``T`` is the post-update cache length. The
    absorbed path wins while::

        L < r*d / (r*d/T + 2*r - d)

    With ``cache_len`` omitted this returns the ``T -> inf`` limit,
    ``r*d / (2*r - d)``, which is the right answer once the cache is much
    larger than the latent. Pass ``cache_len`` for the exact bound: on a cold
    cache, where ``T == L``, it correctly falls below ``L``, because
    materializing K and V is cheaper there for every current model.
    """
    d = qk_nope_head_dim + v_head_dim
    denom = 2 * kv_lora_rank - d
    if cache_len is not None and cache_len > 0:
        denom += kv_lora_rank * d / cache_len
    if denom <= 0:
        return 1
    return max(1, int(kv_lora_rank * d / denom))


def latent_length(kv_latent) -> int:
    """Length of the cached latent, tolerating the quantized 3-tuple form."""
    arr = kv_latent[0] if isinstance(kv_latent, tuple) else kv_latent
    return arr.shape[-2]


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
