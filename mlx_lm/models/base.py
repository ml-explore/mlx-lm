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
    B, seq_len, num_heads, head_dim = hidden_states.shape
    chunk_size = window_overlap * 2
    step = window_overlap
    n_chunks = max((seq_len - window_overlap) // step, 1)

    chunks = mx.zeros((B, n_chunks, chunk_size, num_heads, head_dim), dtype=hidden_states.dtype)

    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_size, seq_len)
        actual_len = end - start
        chunks[:, i, :actual_len, :, :] = hidden_states[:, start:end, :, :]

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
    Computes sliding window attention using efficient chunking.
    
    Args:
        queries: Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        keys: Key tensor of shape [batch_size, num_kv_heads, seq_len_k, head_dim]
        values: Value tensor of shape [batch_size, num_kv_heads, seq_len_v, head_dim]
        scale: Scaling factor for the attention scores
        mask: Optional mask tensor
        sliding_window: Size of the attention window (must be odd)
        
    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    """
    B, num_heads, seq_len, head_dim = queries.shape
    _, num_kv_heads, _, _ = keys.shape

    # Adjust sliding window size
    sliding_window = min(sliding_window, seq_len)
    if sliding_window % 2 == 0:
        sliding_window += 1
    window_overlap = sliding_window // 2

    # First transpose to [B, seq_len, num_heads, head_dim] for chunking
    queries_t = queries.transpose(0, 2, 1, 3)  # [B, seq_len, num_heads, head_dim]
    keys_t = keys.transpose(0, 2, 1, 3)        # [B, seq_len, num_kv_heads, head_dim]
    values_t = values.transpose(0, 2, 1, 3)    # [B, seq_len, num_kv_heads, head_dim]
    
    # Chunk the sequences
    query_chunks = _chunk(queries_t, window_overlap)  # [B, n_chunks, chunk_size, num_heads, head_dim]
    key_chunks = _chunk(keys_t, window_overlap)       # [B, n_chunks, chunk_size, num_kv_heads, head_dim]
    value_chunks = _chunk(values_t, window_overlap)   # [B, n_chunks, chunk_size, num_kv_heads, head_dim]
    
    # Handle num_heads != num_kv_heads case
    if num_heads != num_kv_heads:
        assert num_heads % num_kv_heads == 0
        repeat_factor = num_heads // num_kv_heads
        key_chunks = mx.repeat(key_chunks, repeat_factor, axis=3)
        value_chunks = mx.repeat(value_chunks, repeat_factor, axis=3)
    
    # Apply scale to queries
    query_chunks_scaled = query_chunks * scale
    
    # Compute attention scores
    attention_scores = mx.einsum("bcqhd,bckhd->bcqhk", query_chunks_scaled, key_chunks)
    
    # Apply mask if provided
    if mask is not None:
        # Ensure mask has the right shape for chunking
        if mask.ndim == 2:  # [B, seq_len]
            mask = mask[:, :, None, None]  # [B, seq_len, 1, 1]
        mask_chunks = _chunk(mask, window_overlap)
        attention_scores = attention_scores + mask_chunks
    
    # Compute attention probabilities
    attention_probs = mx.softmax(attention_scores, axis=-1)
    
    # Apply attention to values
    attention_output = mx.einsum("bcqhk,bckhd->bcqhd", attention_probs, value_chunks)
    
    # Recombine chunks
    B_out = attention_output.shape[0]
    n_chunks = attention_output.shape[1]
    chunk_size = attention_output.shape[2]
    
    # Create output tensor and count tensor for averaging
    output = mx.zeros((B_out, seq_len, num_heads, head_dim), dtype=queries.dtype)
    counts = mx.zeros((seq_len,), dtype=mx.float32)
    
    # Accumulate chunks into output
    for i in range(n_chunks):
        start_idx = i * window_overlap
        end_idx = min(start_idx + chunk_size, seq_len)
        chunk_len = end_idx - start_idx
        
        # Add the chunk to the output
        output[:, start_idx:end_idx, :, :] += attention_output[:, i, :chunk_len, :, :]
        counts[start_idx:end_idx] += 1.0
    
    # Normalize by counts (avoiding division by zero)
    counts = mx.maximum(counts, 1.0)
    output = output / counts.reshape(1, -1, 1, 1)
    
    # Transpose back to original format [B, num_heads, seq_len, head_dim]
    return output.transpose(0, 2, 1, 3)



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
