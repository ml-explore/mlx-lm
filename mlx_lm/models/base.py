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


def create_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    offset: int = 0,
    lengths: Optional[mx.array] = None,
):
    """
    Creates a causal mask with sliding window constraint.
    
    Args:
        seq_len: Length of the sequence
        sliding_window: Size of the sliding window
        offset: Offset for the sequence (for caching)
        lengths: Optional tensor of sequence lengths for padding mask
        
    Returns:
        A boolean mask of shape [seq_len, seq_len] or [B, seq_len, seq_len] if lengths provided
    """
    # Ensure sliding window is odd
    if sliding_window % 2 == 0:
        sliding_window += 1
    
    # Create position indices
    row_indices = mx.arange(offset, offset + seq_len)[:, None]  # [seq_len, 1]
    col_indices = mx.arange(offset + seq_len)[None, :]         # [1, seq_len]
    
    # Causal mask: can only attend to positions up to current position
    causal_mask = row_indices >= col_indices
    
    # Window constraint: can only attend to positions within sliding window
    window_mask = row_indices >= (col_indices - sliding_window + 1)
    
    # Combine masks: must satisfy both causal and window constraints
    mask = causal_mask & window_mask
    
    # Apply sequence length mask if provided
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
    
    print(mask.shape)
    
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


def create_sliding_window_causal_mask(seq_len: int, window_size: int, batch_size: int = 1):
    """
    Creates a causal mask with sliding window attention for transformer models.
    
    Args:
        seq_len: Length of the sequence
        window_size: Size of the attention window (will be made odd if even)
        batch_size: Batch size
        
    Returns:
        A mask tensor of shape [batch_size, 1, seq_len, seq_len] where:
        - 0.0 means a token can attend to another token
        - -inf (or a large negative value) means a token cannot attend to another token
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Create a causal mask (upper triangular)
    indices = mx.arange(seq_len)
    causal_mask = indices[:, None] < indices[None, :]  # Shape: [seq_len, seq_len]
    
    # Create a window mask to limit attention to nearby tokens
    window_mask = mx.abs(indices[:, None] - indices[None, :]) > window_size // 2
    
    # Combine masks: a position is masked (True) if it's in the future OR outside the window
    combined_mask = causal_mask | window_mask
    
    # Convert boolean mask to float mask (-inf for masked positions, 0 for unmasked)
    float_mask = mx.zeros((seq_len, seq_len), dtype=mx.float32)
    # Use large negative value for masked positions
    neg_inf = mx.array(-1e9, dtype=mx.float32)
    float_mask = mx.where(combined_mask, neg_inf, float_mask)
    
    # Reshape to [1, 1, seq_len, seq_len]
    float_mask = mx.expand_dims(float_mask, axis=0)
    float_mask = mx.expand_dims(float_mask, axis=0)
    
    # Expand for batch dimension if needed
    if batch_size > 1:
        float_mask = mx.broadcast_to(float_mask, (batch_size, 1, seq_len, seq_len))
    
    return float_mask


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
                return create_sliding_window_causal_mask(T, sliding_window)
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


def _chunk_bshd(hidden_states, window_overlap):
    """
    Chunks input tensor of shape [B, H, L, D] into overlapping chunks.
    Returns [B, H, n_chunks, chunk_size, D].
    """
    B, H, L, D = hidden_states.shape
    chunk_size = window_overlap * 2
    step = window_overlap
    n_chunks = max((L - window_overlap + step - 1) // step, 1)
    chunks = mx.zeros((B, H, n_chunks, chunk_size, D), dtype=hidden_states.dtype)
    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_size, L)
        actual_chunk_len = end - start
        chunks[:, :, i, :actual_chunk_len, :] = hidden_states[:, :, start:end, :]
    return chunks


def sliding_window_scaled_dot_product_attention(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: Optional[float],
        mask: Optional[mx.array],
        sliding_window: int,
        cache: Optional[Any] = None):
    """
    Sliding window attention implementation for mlx arrays.

    Inputs:
      queries: [B, num_heads, seq_len, head_dim]
      keys: [B, num_kv_heads, seq_len, head_dim]
      values: [B, num_kv_heads, seq_len, head_dim]
      mask: [B, seq_len] or None
    """
    B, num_heads, seq_len, head_dim = queries.shape
    _, num_kv_heads, _, _ = keys.shape

    if scale is None:
        scale = 1 / math.sqrt(head_dim)

    # Adjust sliding window size and overlap
    sliding_window = min(sliding_window, seq_len)
    if sliding_window % 2 == 0:
        sliding_window += 1
    window_overlap = sliding_window // 2
    chunk_size = window_overlap * 2

    # Chunk q,k,v
    query_chunks = _chunk_bshd(queries, window_overlap)  # [B,H,n_chunks,chunk,dim]
    key_chunks = _chunk_bshd(keys, window_overlap)       # [B,H_kv,n_chunks,chunk,dim]
    value_chunks = _chunk_bshd(values, window_overlap)   # [B,H_kv,n_chunks,chunk,dim]

    # Handle num_heads != num_kv_heads case:
    if num_heads != num_kv_heads:
        assert num_heads % num_kv_heads == 0
        repeat_factor = num_heads // num_kv_heads
        key_chunks = mx.repeat(key_chunks, repeats=repeat_factor, axis=1)
        value_chunks = mx.repeat(value_chunks, repeats=repeat_factor, axis=1)

    # Scaled attention scores
    query_chunks_scaled = query_chunks * scale
    attention_scores = mx.einsum("bhcqd,bhckd->bhcqk", query_chunks_scaled, key_chunks)

    # Apply mask if provided
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, None, :, None]  # (B,1,seq_len,1) broadcastable
        mask_chunks = _chunk_bshd(mask, window_overlap)  # [B,1,n_chunks,chunk,1]
        attention_scores += mask_chunks

    attention_probs = mx.softmax(attention_scores, axis=-1)
    
    # Compute attention outputs
    attention_output = mx.einsum("bhcqk,bhckd->bhcqd", attention_probs, value_chunks) # [B,H,n_chunks,chunk,D]

    # Initialize recombined output tensor and count tensor for averaging overlap
    output = mx.zeros((B, num_heads, seq_len, head_dim), dtype=queries.dtype)
    counts = mx.zeros((seq_len,), dtype=mx.float32)

    n_chunks = attention_output.shape[2]
    for i in range(n_chunks):
        start_idx = i * window_overlap
        end_idx = min(start_idx + chunk_size, seq_len)
        actual_chunk_len = end_idx - start_idx

        # Accumulate and average overlapping chunks
        output[:, :, start_idx:end_idx, :] += attention_output[:, :, i, :actual_chunk_len, :]
        counts[start_idx:end_idx] += 1.0

    # Avoid division by zero, normalize overlaps
    counts = mx.maximum(counts, 1.0)
    output /= counts.reshape(1, 1, -1, 1)

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
                queries=queries, keys=keys, values=values, scale=scale, mask=mask, sliding_window=sliding_window
            )
        else:
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )
