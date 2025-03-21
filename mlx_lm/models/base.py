# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import math
import mlx.core as mx
from mlx.utils import tree_map

from .cache import QuantizedKVCache


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
        mask = mask & (linds <= rinds + window_size)
    if lengths is not None:
        lengths = lengths[:, None, None, None]
        mask = mask & (rinds < lengths)
    return mask


def create_attention_mask(
    h: mx.array, cache: Optional[Any] = None, return_array: bool = False
):
    T = h.shape[1]
    if T > 1:
        offset = 0
        window_size = None
        if cache is not None and cache[0] is not None:
            c = cache[0]
            offset = c.offset
            if hasattr(c, "max_size"):
                window_size = c.max_size
                offset = min(window_size, offset)
                return_array = return_array or offset + T > window_size
        if return_array:
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


def _chunk(hidden_states, window_overlap):
    """Convert into overlapping chunks. Chunk size = 2 * window_overlap, overlap = window_overlap"""
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape

    # Adjust window_overlap to be reasonable
    window_overlap = min(window_overlap, seq_len // 2)
    window_overlap = max(window_overlap, 1)

    chunk_size = 2 * window_overlap
    step = window_overlap
    n_chunks = max((seq_len - window_overlap) // step, 1)

    chunk_shape = (batch_size, n_chunks, chunk_size, num_heads, head_dim)
    chunks = mx.zeros(chunk_shape, dtype=hidden_states.dtype)

    for i in range(n_chunks):
        start_idx = i * step
        end_idx = min(start_idx + chunk_size, seq_len)
        actual_chunk_len = end_idx - start_idx

        chunk_data = hidden_states[:, start_idx:end_idx, :, :]
        
        # Assign directly (no `.at[...].update()` method)
        chunks[:, i, :actual_chunk_len, :, :] = chunk_data

    return chunks

def sliding_window_scaled_dot_product_attention(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: Optional[mx.array],
        sliding_window: int
    ):
    """
    Computes sliding window attention using efficient chunking. adapted from https://amaarora.github.io/posts/2024-07-04%20SWA.html#sliding-window-attention-in-pytorch
    
    Args:
        queries: Query tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
        keys: Key tensor of shape [batch_size, seq_len_k, num_heads, head_dim]
        values: Value tensor of shape [batch_size, seq_len_v, num_heads, head_dim]
        scale: Scaling factor for the attention scores (default: 1/sqrt(head_dim))
        mask: Optional mask tensor
        sliding_window: Size of the attention window (must be odd)
        
    Returns:
        Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = queries.shape
    
    # Default scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Automatically adjust sliding window size
    if sliding_window > seq_len:
        sliding_window = seq_len if seq_len % 2 == 1 else seq_len - 1

    # Ensure sliding_window is odd
    if sliding_window % 2 == 0:
        sliding_window += 1

    window_overlap = sliding_window // 2

    # Debugging print
    print(f"seq_len: {seq_len}, window_overlap: {window_overlap}, n_chunks: {seq_len // window_overlap - 1}")

    # Chunk queries, keys, and values
    query_chunks = _chunk(queries, window_overlap)
    key_chunks = _chunk(keys, window_overlap)
    value_chunks = _chunk(values, window_overlap)

    query_chunks *= scale

    # Compute attention scores
    attention_scores = mx.einsum("bcqhd,bckhd->bcqhk", query_chunks, key_chunks) * scale

    if mask is not None:
        chunked_mask = _chunk(mask[:, :, None, :], window_overlap)  # Ensure correct shape
        attention_scores += chunked_mask

    # Apply mask if needed
    if mask is not None:
        attention_scores += _chunk(mask[:, :, None, None], window_overlap)

    # Softmax
    attention_probs = mx.softmax(attention_scores, axis=-1)

    # Compute output
    attention_output = mx.einsum("bcqhk,bckhd->bcqhd", attention_probs, value_chunks)

    return attention_output.reshape(batch_size, seq_len, num_heads, head_dim)


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
            return sliding_window_scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask, sliding_window=sliding_window
            )
        else:
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )
