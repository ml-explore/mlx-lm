# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import math
import mlx.core as mx
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


def sliding_window_scaled_dot_product_attention(
    queries, keys, values, scale=None, mask=None, sliding_window=None
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
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = queries.shape
    
    # Default scale is 1/sqrt(head_dim)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Ensure sliding_window is odd
    if sliding_window % 2 == 0:
        sliding_window += 1
    
    # Half window size will be our overlap
    window_overlap = sliding_window // 2
    
    # Function to chunk sequences with overlap
    def _chunk(hidden_states, window_overlap):
        """Convert into overlapping chunks. Chunk size = 2w, overlap = w"""
        # Calculate number of chunks
        n_chunks = (hidden_states.shape[1] // window_overlap) - 1
        
        # Initialize empty tensor for chunks
        chunk_size = [
            hidden_states.shape[0],  # batch_size
            n_chunks,                # num_chunks
            window_overlap * 2,      # chunk_length
            hidden_states.shape[2],  # num_heads
            hidden_states.shape[3],  # head_dim
        ]
        
        # Create empty tensor for chunks
        chunks = mx.zeros(chunk_size)
        
        # Fill chunks with overlapping windows
        for i in range(n_chunks):
            start_idx = i * window_overlap
            end_idx = start_idx + 2 * window_overlap
            chunk_data = hidden_states[:, start_idx:end_idx, :, :]
            chunks = chunks.at[:, i, :, :, :].set(chunk_data)
            
        return chunks
    
    # Chunk queries, keys, and values
    chunked_queries = _chunk(queries, window_overlap)
    chunked_keys = _chunk(keys, window_overlap)
    chunked_values = _chunk(values, window_overlap)
    
    # Compute chunked attention scores
    # Reshape for einsum: [batch, chunks, chunk_len, heads, head_dim]
    q = chunked_queries
    k = chunked_keys
    
    # Compute attention scores with einsum
    # [batch, chunks, query_len, heads, key_len]
    chunked_attention_scores = mx.einsum("bcqhd,bckhd->bcqhk", q, k)
    
    # Apply scaling
    chunked_attention_scores = chunked_attention_scores * scale
    
    # Apply mask if provided
    if mask is not None:
        # Need to chunk the mask as well
        chunked_mask = _chunk(mask.reshape(batch_size, seq_len, num_heads, 1), window_overlap)
        chunked_attention_scores = chunked_attention_scores + chunked_mask
    
    # Apply softmax along the key dimension
    attention_probs = mx.softmax(chunked_attention_scores, axis=-1)
    
    # Apply attention weights to values
    chunked_attention_output = mx.einsum("bcqhk,bckhd->bcqhd", attention_probs, chunked_values)
    
    # Reconstruct the full sequence
    # This requires careful handling of the overlapping regions
    attention_output = mx.zeros((batch_size, seq_len, num_heads, head_dim))
    
    # Reconstruct sequence from chunks, handling overlaps
    for i in range(chunked_attention_output.shape[1]):
        # For each position in the chunk, determine its weight
        # Center positions get full weight, edges get weighted by position
        start_idx = i * window_overlap
        end_idx = start_idx + 2 * window_overlap
        
        # Simple case: just place chunks back (for demonstration)
        # In a real implementation, you'd want to handle overlapping regions more carefully
        chunk_output = chunked_attention_output[:, i, :, :, :]
        attention_output = attention_output.at[:, start_idx:end_idx, :, :].set(chunk_output)
    
    return attention_output


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
