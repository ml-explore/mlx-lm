# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    lengths: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if lengths is not None:
        lengths = lengths[:, None, None, None]
        mask = mask & (rinds < lengths)
    return mask


def create_attention_mask(
    h: mx.array,
    cache: Optional[Any] = None,
    window_size: Optional[int] = None,
    return_array: bool = False,
):
    T = h.shape[1]
    if T > 1:
        offset = 0
        # window_size = None
        mask = "causal"
        if cache is not None and cache[0] is not None:
            c = cache[0]
            offset = c.offset
            if hasattr(c, "make_mask"):
                mask = c.make_mask(T, window_size=window_size)
        if return_array and isinstance(mask, str):
            return create_causal_mask(T, offset)
        else:
            return mask
    else:
        if (
            cache is not None
            and cache[0] is not None
            and hasattr(cache[0], "make_mask")
        ):
            return cache[0].make_mask(T, window_size=window_size)
        return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    if hasattr(cache, "bits"):
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
