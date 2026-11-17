# Copyright © 2026 Apple Inc.

from .base import create_attention_mask
from .cache import KVCache, RotatingKVCache


def prefill_query_lengths(length, layer_types, window_size, cache):
    """Return exact query suffix lengths, or None for unsupported caches."""
    lengths = [length] * len(layer_types)
    if not cache or len(cache) != len(layer_types):
        return None
    for layer_type, c in zip(layer_types, cache):
        if layer_type == "full_attention":
            if type(c) is not KVCache:
                return None
        elif layer_type == "sliding_attention":
            if (
                not window_size
                or window_size <= 1
                or type(c) is not RotatingKVCache
                or c.keep != 0
                or c.max_size != window_size
            ):
                return None
        else:
            return None

    if len({c.offset for c in cache}) != 1:
        return None

    needed = 1
    # Each downstream SWA layer needs another window of input rows.
    # A full-attention layer still projects keys and values for the whole chunk.
    for i in range(len(layer_types) - 1, -1, -1):
        lengths[i] = min(length, needed)
        if layer_types[i] == "full_attention":
            break
        needed += window_size - 1
    return lengths


def prefill_layers(h, layers, layer_types, window_size, cache):
    """Fill caches with exact suffixes when prompt logits are not needed."""
    length = h.shape[1]
    query_lengths = prefill_query_lengths(length, layer_types, window_size, cache)
    if query_lengths is None:
        return None

    masks = {}
    for layer, c, layer_type, query_length in zip(
        layers, cache, layer_types, query_lengths
    ):
        key = (layer_type, h.shape[1], c.offset)
        if key not in masks:
            masks[key] = create_attention_mask(
                h,
                c,
                window_size=window_size if layer_type == "sliding_attention" else None,
            )
        h = layer(h, masks[key], c, query_length, length - h.shape[1])
    return h
