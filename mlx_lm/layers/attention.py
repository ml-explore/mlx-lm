# Copyright © 2025 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from mlx_lm.kernels import sparse_mla

__all__ = [
    "MultiLinear",
    "QuantizedMultiLinear",
    "SPARSE_QUERY_BLOCK",
    "blocked_topk",
    "create_attention_mask",
    "create_causal_mask",
    "fused_sparse_attention",
    "min_context",
    "quantized_scaled_dot_product_attention",
    "scaled_dot_product_attention",
    "sparse_attention",
]


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


SPARSE_QUERY_BLOCK = 1024


def min_context(attn, q_pe, kv_latent, k_pe):
    # Cache length from which prefill takes the sparse path. With the fused kernel it is
    # faster than dense attention in our measurements as soon as the cache holds more than
    # index_topk keys (when the indexer starts selecting).
    if (
        attn.sparse_prefill_fused
        and not attn.training
        and sparse_mla.supports(q_pe[0], kv_latent[0, 0], k_pe[0, 0])
    ):
        return attn.sparse_prefill_fused_min_context
    return attn.sparse_prefill_min_context


def fused_sparse_attention(attn, q_nope, q_pe, kv_latent, k_pe, topk, mask, B, L):
    # sparse_attention in one Metal kernel that reads the selected rows by index,
    # or None when the kernel doesn't support the inputs.
    kv, kp = kv_latent[0, 0], k_pe[0, 0]
    if not sparse_mla.supports(q_pe[0], kv, kp):
        return None
    idx = topk[0, 0].astype(mx.int32)  # [L, K]
    if mask is not None:
        valid = mx.take_along_axis(mask.reshape(L, -1), idx, axis=-1)
        idx = mx.where(valid, idx, -1)  # the kernel skips -1
    q = attn.embed_q(q_nope)[0]  # [H, L, 512]
    out = sparse_mla.sparse_mla_attention(q, q_pe[0], kv, kp, idx, attn.scale)
    if out is None:
        return None
    out = attn.unembed_out(out[None])  # [1, H, L, v_head_dim]
    return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


def sparse_attention(attn, q_nope, q_pe, kv_latent, k_pe, topk, mask, B, L):
    # Absorbed form: 576-wide keys (512 latent + 64 rope), 512-wide values. B is 1.
    if attn.sparse_prefill_fused and not attn.training:
        args = (attn, q_nope, q_pe, kv_latent, k_pe, topk, mask, B, L)
        out = fused_sparse_attention(*args)
        if out is not None:
            return out
    block = SPARSE_QUERY_BLOCK
    if L > block:
        # Query blocks keep the gathered keys small, which is faster for long chunks.
        outs = []
        for s in range(0, L, block):
            e = min(L, s + block)
            m = None if mask is None else mask.reshape(L, -1)[s:e]
            q = (q_nope[:, :, s:e], q_pe[:, :, s:e])
            idx = topk[:, :, s:e]
            outs.append(sparse_attention(attn, *q, kv_latent, k_pe, idx, m, B, e - s))
        return mx.concatenate(outs, axis=1)
    q_full = mx.concatenate([attn.embed_q(q_nope), q_pe], axis=-1)[0]  # [H, L, 576]
    kv_full = mx.concatenate([kv_latent, k_pe], axis=-1)[0, 0]  # [N, 576]
    idx = topk[0, 0]  # [L, K]
    gathered = kv_full[idx]  # [L, K, 576], shared by all heads

    q_t = q_full.transpose(1, 0, 2)  # [L, H, 576]
    scores = (q_t @ gathered.swapaxes(-1, -2)).astype(mx.float32) * attn.scale
    if mask is not None:
        valid = mx.take_along_axis(mask.reshape(L, -1), idx, axis=-1)  # [L, K]
        scores = mx.where(valid[:, None, :], scores, -mx.inf)
    probs = mx.softmax(scores, axis=-1).astype(gathered.dtype)
    latent_out = probs @ gathered[..., : attn.kv_lora_rank]  # [L, H, 512]
    latent_out = latent_out.transpose(1, 0, 2)[None]  # [1, H, L, 512]
    out = attn.unembed_out(latent_out)  # [1, H, L, v_head_dim]
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(out)


def blocked_topk(indexer, x, qr, mask, cache, block=4096):
    # Stock indexer scores, one key block at a time, without the [b, H, s, N] tensor.
    # Heads are added in the stock order. A running top-k keeps the sorts short and
    # selects the same keys as stock (up to the order of equal scores).
    b, s, _ = x.shape
    q = indexer.wq_b(qr)
    q = q.reshape(b, s, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
    k = indexer.k_norm(indexer.wk(x))
    k = mx.reshape(k, (b, 1, s, indexer.head_dim))
    offset = cache.offset if cache is not None else 0
    q = indexer.rope(q, offset=offset)
    k = indexer.rope(k, offset=offset)
    if cache is not None:
        k, _ = cache.update_and_fetch(k, k)
        cache.values = mx.zeros_like(cache.keys)
    n = k.shape[2]
    if n <= indexer.index_topk:
        return None

    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5 * indexer.softmax_scale)
    weights = weights.swapaxes(-1, -2)[..., None]  # [b, H, s, 1]

    best_v = best_i = None
    for s0 in range(0, n, block):
        s1 = min(n, s0 + block)
        sc = mx.maximum(q @ k[:, :, s0:s1].swapaxes(-1, -2), 0) * weights
        sc = sum(sc[:, h : h + 1] for h in range(sc.shape[1]))  # [b, 1, s, blk]
        if mask is not None:
            sc = mx.where(mask[..., s0:s1], sc, -float("inf"))
        ids = mx.broadcast_to(mx.arange(s0, s1), sc.shape)
        if best_v is not None:
            sc = mx.concatenate([best_v, sc], axis=-1)
            ids = mx.concatenate([best_i, ids], axis=-1)
        k_top = indexer.index_topk
        pos = mx.stop_gradient(mx.argpartition(sc, kth=-k_top, axis=-1)[..., -k_top:])
        best_v = mx.take_along_axis(sc, pos, axis=-1)
        best_i = mx.take_along_axis(ids, pos, axis=-1)
    return best_i
