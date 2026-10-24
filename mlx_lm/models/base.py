# Copyright © 2023 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Optional

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


# Query rows per band, and the key-tile multiple band starts snap to. The fused
# kernel's key tile (16 or 32) and query tile (32 or 64) divide these, so a band
# sees the same tile grid and online-softmax order as the single call.
_BAND_QUERIES = 256
_BAND_KEY_ALIGN = 64
# Other head dims run the unfused fallback, whose softmax reduction depends on the key count.
_BAND_HEAD_DIMS = (64, 128)
# Queries <= 8 take the sdpa_vector kernel, which accumulates in another order.
_SDPA_VECTOR_MAX_QUERIES = 8


def _banded_sliding_attention(queries, keys, values, cache, scale, mask, sinks, window):
    """Sliding-window prefill attention that skips key tiles the window never reaches.

    A rotating cache hands back ``window - 1 + L`` keys for an ``L``-token chunk,
    and a full-width array mask makes the fused kernel visit all of them for every
    query row. Fully masked key tiles add exactly zero to the online softmax, so
    processing query bands over only the keys they can reach is bit-identical.
    Runs only when the caller passes ``window`` equal to ``cache.max_size``, which
    states that ``mask`` is that cache's own sliding-window mask: the mask's shape
    alone cannot tell a sliding window from any other ``[L, kL]`` mask. Returns None
    when the call does not fit that pattern.
    """
    from .cache import RotatingKVCache

    L, kL = queries.shape[2], keys.shape[2]
    if (
        not isinstance(cache, RotatingKVCache)
        or window != cache.max_size
        or cache.keep != 0
        or not isinstance(mask, mx.array)
        or mask.shape != (L, kL)
        or queries.shape[-1] not in _BAND_HEAD_DIMS
        or L <= _SDPA_VECTOR_MAX_QUERIES
        or 0 < L % _BAND_QUERIES <= _SDPA_VECTOR_MAX_QUERIES
        or kL <= _BAND_QUERIES + cache.max_size
    ):
        return None
    prior = kL - L
    outs = []
    for s in range(0, L, _BAND_QUERIES):
        n = min(_BAND_QUERIES, L - s)
        start = max(0, prior + s - cache.max_size + 1)
        start -= start % _BAND_KEY_ALIGN
        stop = prior + s + n
        outs.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, s : s + n],
                keys[:, :, start:stop],
                values[:, :, start:stop],
                scale=scale,
                mask=mask[s : s + n, start:stop],
                sinks=sinks,
            )
        )
    return mx.concatenate(outs, axis=2)


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
    sliding_window: Optional[int] = None,
) -> mx.array:
    """``sliding_window``: set it only when ``mask`` is ``cache``'s own sliding-window
    mask of that width, to let prefill skip the key tiles the window never reaches."""
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
        banded = _banded_sliding_attention(
            queries, keys, values, cache, scale, mask, sinks, sliding_window
        )
        if banded is not None:
            return banded
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )


def gather_last_axis(x: mx.array, group: mx.distributed.Group) -> mx.array:
    """Gather the last axis of ``x``, which is split across ``group``."""
    n, size = group.size(), x.shape[-1]
    parts = mx.distributed.all_gather(x.reshape(-1, size), group=group)
    # (ranks * rows, size) to (rows, ranks * size), in rank order.
    parts = parts.reshape(n, -1, size).transpose(1, 0, 2)
    return parts.reshape(*x.shape[:-1], n * size)
