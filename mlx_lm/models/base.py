# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from cache import QuantizedKVCache


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


def create_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    offset: int = 0,
    lengths: Optional[mx.array] = None
):
    if sliding_window % 2 == 0:
        sliding_window += 1

    row_indices = mx.arange(offset, offset + seq_len)[:, None]
    col_indices = mx.arange(offset + seq_len)[None, :]

    causal_mask = row_indices >= col_indices
    window_mask = row_indices >= (col_indices - sliding_window + 1)
    mask = causal_mask & window_mask
    if lengths is not None:
        # lengths: [B]
        batch_size = lengths.shape[0]
        # Reshape to [B, 1, 1]
        lengths = lengths[:, None, None]
        # Expand col_indices to [1, 1, seq_len]
        expanded_col_indices = col_indices[None, :, :]
        # Create padding mask: [B, 1, seq_len]
        padding_mask = expanded_col_indices < lengths
        # Expand mask to [B, seq_len, seq_len]
        mask = mask[None, :, :] & padding_mask
    return mask


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
        mask = mask & (linds <= rinds + window_size)
    if lengths is not None:
        lengths = lengths[:, None, None, None]
        mask = mask & (rinds < lengths)
    return mask


def create_attention_mask(
    h: mx.array, 
    cache: Optional[Any] = None, 
    return_array: bool = False,
    sliding_window: Optional[int] = None
):
    T = h.shape[1]
    
    if T > 1:
        offset = 0
        window_size = sliding_window
        
        if cache is not None and cache[0] is not None:
            c = cache[0]
            offset = c.offset
            if hasattr(c, "max_size") and sliding_window is None:
                window_size = c.max_size
                offset = min(window_size, offset)
                return_array = return_array or offset + T > window_size
        
        if return_array or sliding_window is not None:
            if sliding_window is not None:
                return create_sliding_window_mask(T, sliding_window)
            else:
                return create_causal_mask(T, offset, window_size=window_size)
        else:
            return "causal"
    else:
        mask = None
    
    return mask


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
        scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def sliding_window_attention(queries, keys, values, scale: int, mask=None, sliding_window=512):
    """
    Computes sliding window self-attention in MLX.

    Supports both Multi-Head Attention (MHA) and Grouped Query Attention (GQA).

    Args:
        q: Query tensor of shape (B, Hq, L, D)
        k: Key tensor of shape (B, Hkv, L, D)
        v: Value tensor of shape (B, Hkv, L, D)
        mask: Optional causal mask of shape (B, L)
        window_size: The local attention window size.

    Returns:
        attn_output: Tensor of shape (B, Hq, L, D)
    """
    B, Hq, L, D = queries.shape
    Hkv = keys.shape[1]  # Number of KV heads

    queries *= scale

    # If using GQA (Hq > Hkv), expand KV heads
    if Hq != Hkv:
        factor = Hq // Hkv
        keys = mx.repeat(keys, factor, axis=1)  # (B, Hq, L, D)
        values = mx.repeat(values, factor, axis=1)  # (B, Hq, L, D)

    # Compute scaled dot-product attention
    scores = mx.einsum('bhid,bhjd->bhij', queries, keys)

    # Apply sliding window mask
    idx = mx.arange(L)
    local_mask = mx.abs(idx[:, None] - idx[None, :]) > sliding_window
    scores = mx.where(local_mask[None, None, :, :], -mx.inf, scores)

    # Apply causal mask if provided
    if mask is not None:
        scores += mask

    # Compute attention weights
    attn_weights = mx.softmax(scores, axis=-1)

    # Compute attention output
    attn_output = mx.einsum('bhij,bhjd->bhid', attn_weights, values)
    return attn_output


def quantized_sliding_window_attention(queries, keys, values, scale: int, mask=None, sliding_window=512):
    B, Hq, L, D = queries.shape
    Hkv = keys.shape[1]  # Number of KV heads

    queries *= scale

    # If using GQA (Hq > Hkv), expand KV heads
    if Hq != Hkv:
        factor = Hq // Hkv
        keys = mx.repeat(keys, factor, axis=1)  # (B, Hq, L, D)
        values = mx.repeat(values, factor, axis=1)  # (B, Hq, L, D)

    # Reshape for quantized matmul
    q_flat = mx.reshape(queries, (B * Hq, L, D))
    k_flat = mx.reshape(keys, (B * Hq, L, D))
    v_flat = mx.reshape(values, (B * Hq, L, D))
    
    # Transpose k for matrix multiplication
    k_flat_t = mx.transpose(k_flat, (0, 2, 1))  # (B*Hq, D, L)
    
    # QK attention using quantized matmul
    scores = mx.quantized_matmul(q_flat, k_flat_t)
    scores = mx.reshape(scores, (B, Hq, L, L))
    
    # Apply sliding window mask
    idx = mx.arange(L)
    local_mask = mx.abs(idx[:, None] - idx[None, :]) > sliding_window
    scores = mx.where(local_mask[None, None, :, :], -mx.inf, scores)

    # Apply causal mask if provided
    if mask is not None:
        scores += mask

    # Compute attention weights
    attn_weights = mx.softmax(scores, axis=-1)  # (B, Hq, L, L)
    
    # Reshape attention weights for second matmul
    attn_flat = mx.reshape(attn_weights, (B * Hq, L, L))
    
    # Compute attention outputs using quantized matmul
    output = mx.quantized_matmul(attn_flat, v_flat)
    output = mx.reshape(output, (B, Hq, L, D))
    
    return output


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sliding_window: Optional[int] = None
) -> mx.array:
    if isinstance(cache, QuantizedKVCache):
        if sliding_window is not None:
            return quantized_sliding_window_attention(
                queries=queries, keys=keys, values=values, scale=scale, mask=mask, sliding_window=sliding_window
            )
        else:
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
        if sliding_window is not None:
            return sliding_window_attention(
                queries=queries, keys=keys, values=values, scale=scale, mask=mask, sliding_window=sliding_window
            )
        else:
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )
