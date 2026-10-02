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


# Cap on the attention-scores allocation in :func:`quantized_scaled_dot_product_attention`,
# in bytes. The scores are [B, n_kv, n_rep, T, K] in fp16 or fp32 (fp32 for any non-fp16
# query — this model's bf16 weights, verified), so with a 4 GiB budget the query axis is
# tiled to at most ~4 GiB / (n_q_heads * K * bytes) rows per pass — 4 GiB holds a 2048-row
# pass to K ~ 14k at 24 q heads/fp32 and a 512-row pass to K ~ 57k, comfortably covering
# the ~262k-token context ceiling. Verified at the actual crash shape (B=1, 24 q / 4 kv
# heads, n_rep=6, T=2048, D=256, 4-bit KV, bf16 queries — the exact qwen3.8-27b geometry:
# untiled scores 30,337,597,440 bytes ≈ 28.16 GiB at K = 154,305, vs the 28.08 GiB
# Metal single-buffer cap -> metal::malloc abort): the tiled path completes in 7 x
# ~291-row passes at 28.77 GiB peak in 1.6 s. Set to 0 to disable tiling (restore the
# single-pass allocation).
QDPA_SCORES_BUDGET_BYTES = 4_294_967_296


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
    K = q_keys[0].shape[-2]

    # mx.quantized_matmul emits fp32 scores for non-fp16 queries (fp16 for fp16 queries),
    # so the per-pass scores allocation is [B, n_kv, n_rep, T, K], growing linearly in the
    # KV length K. On a long-context hybrid with quantized KV the 2048-row prefill chunk
    # hit 28.16 GiB of fp32 scores at K = 154,305 — over the 28.08 GiB Metal
    # single-buffer cap, which aborts the prefill with a metal::malloc error. Tile the
    # QUERIES instead: the row reduction (mask + softmax over the full K) is per-row, so
    # a partition of the query rows is an exact, embarrassingly-parallel split — verified
    # bit-identical to the single pass (max|d| = 0) for the fp16- and bf16-scores paths,
    # the 4D non-GQA branch, and the GQA n_rep>1 branch alike. The budget bounds peak
    # memory for any context length; prompts short enough to fit take the identical
    # single-tile path (tile = L).
    budget = QDPA_SCORES_BUDGET_BYTES
    tile = min(L, max(1, budget // (max(n_kv_heads, 1) * max(n_repeats, 1) * max(K, 1) * 4))) if budget > 0 else L
    if n_repeats > 1:
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    def _align(m, t):
        # Grow rank by inserting size-1 axes before (L, K) so 4D batch masks
        # broadcast against 5D expanded GQA scores (mirrors mlx_vlm #1567 fix).
        while m.ndim < t.ndim:
            m = mx.expand_dims(m, axis=max(m.ndim - 2, 0))
        return m

    out = []
    queries *= scale
    for s in range(0, L, tile):
        e = min(s + tile, L)
        q = queries[..., s:e, :]
        if n_repeats > 1:
            q = mx.reshape(q, (B, n_kv_heads, n_repeats, e - s, D))
        scores = mx.quantized_matmul(
            q, *q_keys, transpose=True, group_size=group_size, bits=bits
        )
        if mask is not None:
            if isinstance(mask, str):
                k_indices = mx.arange(K)
                q_indices = mx.arange(K - L + s, K - L + e)
                tmask = (q_indices[:, None] >= k_indices[None])
            else:
                tmask = mask[..., s:e, :]
            tmask = _align(tmask, scores)
            if tmask.dtype == mx.bool_:
                scores = mx.where(tmask, scores, mx.finfo(scores.dtype).min)
            else:
                scores += tmask
        scores = mx.softmax(scores, axis=-1, precise=True)
        o = mx.quantized_matmul(
            scores, *q_values, transpose=False, group_size=group_size, bits=bits
        )
        if n_repeats > 1:
            o = mx.reshape(o, (B, n_q_heads, e - s, D))
        out.append(o)
    return mx.concatenate(out, axis=-2) if len(out) > 1 else out[0]


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
