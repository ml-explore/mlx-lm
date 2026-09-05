"""Chunked prefill wrapper for spark2_5 (or any mlx-lm model with RotatingKVCache).

The wrapper slices the prompt along the sequence axis and runs the model on
each slab with a shared cache. Between slabs we call mx.eval on the cache
state so RotatingKVCache truncates its SWA layers and MLX can free the
transient activations.

Only the final slab returns logits; internal slabs are called for their
side-effect on the cache.
"""
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def chunked_prefill(model, prompt_ids, cache=None, chunk_size=512):
    """Run prefill over prompt_ids in chunks of chunk_size.

    Args:
        model: mlx-lm model.
        prompt_ids: 1-D python list or mx.array of token ids.
        cache: existing prompt cache, or None to build one.
        chunk_size: tokens per slab.

    Returns:
        (last_logits, cache) where last_logits is [1, 1, vocab] logits for
        the final token of the prompt.
    """
    if cache is None:
        cache = make_prompt_cache(model)
    if isinstance(prompt_ids, list):
        prompt_ids = mx.array(prompt_ids)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None]  # [1, L]
    L = prompt_ids.shape[1]
    i = 0
    last_logits = None
    while i < L:
        j = min(i + chunk_size, L)
        slab = prompt_ids[:, i:j]
        logits = model(slab, cache=cache)
        mx.eval([c.state for c in cache])
        if j == L:
            last_logits = logits[:, -1:, :]
            mx.eval(last_logits)
        else:
            del logits
        mx.clear_cache()
        i = j
    return last_logits, cache
