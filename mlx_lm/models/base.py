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
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
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


def _turbo_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array],
    q_values: tuple[mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    mode: str,
    group_size: int,
) -> mx.array:
    """SDPA for TurboQuant-compressed keys/values.

    Rotates queries with a normalized WHT before calling the fused kernel.
    The kernel handles causal masking, GQA fan-out, online softmax, and
    the full attention reduction in one Metal pass.
    """
    import math

    D = queries.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(D)

    # WHT in float32 to match encoding precision (bfloat16 butterfly accumulates
    # enough error to shift softmax peaks on large-scale models).
    try:
        from .turbo_metal import is_available, wht_rotate_metal
        if is_available():
            q_rot = wht_rotate_metal(queries, scale=inv_sqrt_d)
        else:
            raise ImportError
    except (ImportError, Exception):
        from .turbo_cache import _hadamard_transform
        q_rot = _hadamard_transform(queries.astype(mx.float32), scale=inv_sqrt_d)
        q_rot = q_rot.astype(queries.dtype)

    k_packed, k_scales = q_keys
    v_packed, v_scales = q_values

    causal = mask == "causal" if isinstance(mask, str) else False
    arr_mask = None if (mask is None or isinstance(mask, str)) else mask

    return mx.fast.quantized_scaled_dot_product_attention(
        q_rot,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        scale=scale,
        mask=arr_mask,
        mode=mode,
        group_size=group_size,
        causal=causal,
    )


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    from .turbo_cache import TurboQuantKVCache

    if isinstance(cache, TurboQuantKVCache) and isinstance(keys, tuple):
        if sinks is not None:
            raise ValueError("TurboQuant SDPA does not support attention sinks.")
        return _turbo_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            mode=cache.mode,
            group_size=cache.group_size,
        )
    elif hasattr(cache, "bits") and not isinstance(cache, TurboQuantKVCache):
        # Standard QuantizedKVCache (affine/mxfp4/…)
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
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
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
