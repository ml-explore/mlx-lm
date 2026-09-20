"""Prepare, save and load a sharded prompt cache (one file per rank).

A corpus is prefilled once with ``ShardedKVCache`` (each rank keeps only its
own slice of K/V). ``save_sharded_cache`` writes that slice; later,
``load_sharded_cache`` restores it on the same rank so questions can be asked
without redoing the prefill. Files are named ``<prefix>.rank<r>of<P>.safetensors``.

Only the layers that keep a full, growing KV cache are sharded. Layers with
another kind of cache (for example the sliding-window layers of Gemma 3) keep
their small ordinary cache on every rank; it is saved next to the shard in
``<prefix>.rank<r>of<P>.local.safetensors``.
"""

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .models.cache import KVCache, load_prompt_cache, make_prompt_cache, save_prompt_cache
from .models.sharded_cache import ShardedKVCache


def _path(prefix: str, rank: int, size: int) -> str:
    return f"{prefix}.rank{rank}of{size}.safetensors"


def _local_path(prefix: str, rank: int, size: int) -> str:
    return f"{prefix}.rank{rank}of{size}.local.safetensors"


def make_sharded_cache(
    model, group: Any, kv_bits: Optional[int] = None, group_size: int = 64
) -> List[Any]:
    """The model's own cache list, with every plain ``KVCache`` replaced by a
    ``ShardedKVCache`` on ``group``. Other cache kinds are kept as they are."""
    return [
        ShardedKVCache(group=group, kv_bits=kv_bits, group_size=group_size)
        if type(c) is KVCache
        else c
        for c in make_prompt_cache(model)
    ]


def _backbone(model):
    while not hasattr(model, "model") and hasattr(model, "language_model"):
        model = model.language_model
    return model.model


def save_sharded_cache(
    prefix: str,
    caches: List[Any],
    group: Any,
    total_tokens: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write this rank's shard of every sharded layer. Returns the file path."""
    rank, size = group.rank(), group.size()
    sharded = [i for i, c in enumerate(caches) if isinstance(c, ShardedKVCache)]
    replicated = [i for i, c in enumerate(caches) if not isinstance(c, ShardedKVCache)]
    if not sharded:
        raise ValueError("no sharded layers in the cache")
    first = caches[sharded[0]]
    if first.keys is None:
        raise ValueError(
            f"rank {rank} stores no blocks; use a longer corpus, a smaller block "
            "size, or different shard weights"
        )
    tensors = {}
    for i in sharded:
        c = caches[i]
        if c.kv_bits is not None:
            for j in range(3):
                tensors[f"layer{i}.keys.{j}"] = c.keys[j][..., : c.local_offset, :]
                tensors[f"layer{i}.values.{j}"] = c.values[j][..., : c.local_offset, :]
        else:
            tensors[f"layer{i}.keys"], tensors[f"layer{i}.values"] = (
                c.keys[..., : c.local_offset, :],
                c.values[..., : c.local_offset, :],
            )
    meta = {
        "rank": rank,
        "size": size,
        "n_layers": len(caches),
        "sharded_layers": sharded,
        "replicated_layers": replicated,
        "shard_start": first.shard_start,
        "local_len": first.local_offset,
        "shard_lengths": first.shard_lengths,
        "kv_bits": first.kv_bits,
        "group_size": first.group_size,
        "total_tokens": total_tokens,
        "extra": extra or {},
    }
    path = _path(prefix, rank, size)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mx.save_safetensors(path, tensors, metadata={"meta": json.dumps(meta)})
    if replicated:
        save_prompt_cache(_local_path(prefix, rank, size), [caches[i] for i in replicated])
    return path


def load_sharded_cache(
    prefix: str, group: Any
) -> Tuple[List[Any], Dict[str, Any]]:
    """Restore this rank's caches. The cluster size must match the saved one.

    The last rank owns new tokens (question / answer), the same rule used
    when the corpus was prepared.
    """
    rank, size = group.rank(), group.size()
    tensors, raw = mx.load(_path(prefix, rank, size), return_metadata=True)
    meta = json.loads(raw["meta"])
    if meta["size"] != size or meta["rank"] != rank:
        raise ValueError(
            f"cache was saved for rank {meta['rank']} of {meta['size']}, "
            f"loading on rank {rank} of {size}"
        )
    caches: List[Any] = [None] * meta["n_layers"]
    for i in meta["sharded_layers"]:
        c = ShardedKVCache(
            shard_start=meta["shard_start"],
            owns_new_token=(rank == size - 1),
            shard_lengths=meta["shard_lengths"],
            kv_bits=meta.get("kv_bits"),
            group_size=meta.get("group_size", 64),
            group=group,
        )
        if c.kv_bits is not None:
            c.keys = tuple(tensors[f"layer{i}.keys.{j}"] for j in range(3))
            c.values = tuple(tensors[f"layer{i}.values.{j}"] for j in range(3))
        else:
            c.keys = tensors[f"layer{i}.keys"]
            c.values = tensors[f"layer{i}.values"]
        c.local_offset = meta["local_len"]
        c.query_mode = True
        caches[i] = c
    if meta["replicated_layers"]:
        local = load_prompt_cache(_local_path(prefix, rank, size))
        for i, c in zip(meta["replicated_layers"], local):
            caches[i] = c
    return caches, meta


def block_owners(n_blocks: int, size: int, weights=None) -> List[int]:
    """Which rank stores each block: round-robin, or weighted (smooth) if given."""
    if not weights:
        return [b % size for b in range(n_blocks)]
    total = float(sum(weights))
    credit = [0.0] * size
    owners = []
    for _ in range(n_blocks):
        for i, w in enumerate(weights):
            credit[i] += w
        owner = max(range(size), key=lambda i: credit[i])
        owners.append(owner)
        credit[owner] -= total
    return owners


def block_ranges(total: int, block_size: int) -> List[Tuple[int, int]]:
    """(start, end) of every prefill block."""
    return [(s, min(total, s + block_size)) for s in range(0, total, block_size)]


def prefill_in_blocks(
    model,
    tokens: List[int],
    caches: List[Any],
    group: Any,
    block_size: int,
    owners: List[int],
    start_block: int = 0,
    on_block=None,
):
    """Build a sharded KV cache from ``tokens`` block by block.

    Every block is fed as replicated "question-style" tokens against the cache
    built so far, so temporary memory depends on ``block_size`` only, never on
    the corpus length. The block's K/V are kept (not rolled back) on the rank
    ``owners[b]``; attention over the stored past is split across all ranks.
    Only the backbone runs (no LM head), since only K/V are needed.

    ``on_block(blocks_done, tokens_done)`` is called after each block (for
    progress and checkpoints).
    """
    rank = group.rank()
    sharded = [c for c in caches if isinstance(c, ShardedKVCache)]
    backbone = _backbone(model)
    ranges = block_ranges(len(tokens), block_size)
    if len(owners) < len(ranges):
        raise ValueError(f"{len(ranges)} blocks but only {len(owners)} owners")
    for b in range(start_block, len(ranges)):
        start, end = ranges[b]
        for c in sharded:
            c.query_mode = True
            c.owns_new_token = owners[b] == rank
            c.offset = start
        ids = mx.array(tokens[start:end])[None]
        mx.eval(backbone(ids, caches))
        if on_block is not None:
            on_block(b + 1, end)
    for c in sharded:
        c._query_offset = None


def _snapshot(cache):
    """Independent copy of an ordinary cache's state (MLX arrays are copied)."""
    state = tuple(x * 1 if isinstance(x, mx.array) else x for x in cache.state)
    mx.eval([x for x in state if isinstance(x, mx.array)])
    return state


@contextmanager
def query_scope(caches: List[Any], base: int):
    """Ask something against a loaded cache, then roll everything back.

    Yields ``set_position(pos)``; call it before every forward pass with the
    true global position of the next token (``base`` for the question, then
    ``base + n`` while decoding). On exit the sharded layers drop what the
    question and answer added, and ordinary layers (sliding window) return to
    their saved state, so the prepared corpus stays as it was.
    """
    sharded = [c for c in caches if isinstance(c, ShardedKVCache)]
    marks = [c.local_offset for c in sharded]
    saved = [(c, _snapshot(c)) for c in caches if not isinstance(c, ShardedKVCache)]

    def set_position(pos: int):
        for c in sharded:
            c.offset = pos

    set_position(base)
    try:
        yield set_position
    finally:
        for c, mark in zip(sharded, marks):
            c.trim(c.local_offset - mark)
            c._query_offset = None
        for c, state in saved:
            c.state = state
