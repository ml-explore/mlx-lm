# Copyright © 2023-2024 Apple Inc.

import inspect
import os
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


# Above this many query rows, dequantize K/V and use the fused flash kernel
# instead of the decomposed qmm->softmax->qmm path. The decomposed path
# materializes an (n_heads, L, S) scores matrix to memory, which loses to
# flash attention once L is large (prefill chunks); below the threshold the
# decomposed path wins because it reads the smaller quantized K/V bytes and
# skips the full dequant. Measured crossovers on M5 Max (S=2k..32k, bits 4/8
# identical): L=128 for GQA (n_repeats>=2), L=192 for MHA (n_repeats==1) where
# the fp16 K/V dequant transient is proportionally larger. These are
# M5-measured defaults, not universal constants; MLX_LM_QSDPA_FLASH_MIN_L
# overrides both (0 or negative disables the flash route entirely).
_QSDPA_ENV = os.environ.get("MLX_LM_QSDPA_FLASH_MIN_L")
if _QSDPA_ENV is not None and int(_QSDPA_ENV) <= 0:
    _QUANT_SDPA_FLASH_MIN_L_GQA = _QUANT_SDPA_FLASH_MIN_L_MHA = float("inf")
elif _QSDPA_ENV is not None:
    _QUANT_SDPA_FLASH_MIN_L_GQA = _QUANT_SDPA_FLASH_MIN_L_MHA = int(_QSDPA_ENV)
else:
    _QUANT_SDPA_FLASH_MIN_L_GQA = 128
    _QUANT_SDPA_FLASH_MIN_L_MHA = 192


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

    flash_min_l = (
        _QUANT_SDPA_FLASH_MIN_L_MHA
        if n_repeats == 1
        else _QUANT_SDPA_FLASH_MIN_L_GQA
    )
    if L >= flash_min_l:
        # Large-L (prefill-shaped) case: the transient fp16 K/V costs
        # S * n_kv_heads * D * 4 bytes but avoids the O(L*S) scores
        # round-trip; measured 1.3-2.5x faster than the decomposed path
        # beyond the crossover on all repeat/bits combinations.
        keys = mx.dequantize(*q_keys, group_size=group_size, bits=bits)
        values = mx.dequantize(*q_values, group_size=group_size, bits=bits)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )

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
        if n_repeats > 1 and mask.ndim > 3:
            mask = mx.expand_dims(mask, -3)
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
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
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
