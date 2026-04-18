"""Disk-backed LRU prompt cache for mlx_lm.server.

Wraps LRUPromptCache to persist evicted KV caches to disk and restore
them on cache miss. Survives server restarts — the cache directory
retains all previously-seen prompt caches.

Usage:
    cache = DiskBackedPromptCache(
        max_size=20,
        cache_dir="/tmp/kv_cache",
    )
    # Drop-in replacement for LRUPromptCache
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx

from .models.cache import KVCache, LRUPromptCache, make_prompt_cache

logger = logging.getLogger(__name__)


def _cache_key_hash(model: Any, tokens: List[int]) -> str:
    """Stable hash for a (model, tokens) cache key."""
    raw = f"{model}:{','.join(str(t) for t in tokens)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _save_cache_to_disk(
    cache_dir: Path, model: Any, tokens: List[int],
    prompt_cache: List[Any], cache_type: str = "assistant",
):
    """Save a prompt cache entry to disk atomically.

    Writes to a temp directory first, then renames. If the process
    crashes mid-write, only the temp dir is left (no corrupt entry
    in the main namespace).
    """
    if cache_dir is None:
        return
    h = _cache_key_hash(model, tokens)
    entry_dir = cache_dir / h
    tmp_dir = cache_dir / f".tmp_{h}"

    # Write to temp dir; clean up on any failure
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta = {
            "model": str(model),
            "tokens": tokens,
            "cache_type": cache_type,
            "n_layers": len(prompt_cache),
        }
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        for i, c in enumerate(prompt_cache):
            arrays = {}
            state = c.state if hasattr(c, "state") else []
            if isinstance(state, (list, tuple)):
                for j, arr in enumerate(state):
                    if isinstance(arr, mx.array):
                        arrays[f"s{j}"] = arr
            if arrays:
                mx.savez(str(tmp_dir / f"layer_{i}.npz"), **arrays)

            layer_meta = {
                "class": type(c).__name__,
                "offset": getattr(c, "offset", 0),
            }
            if hasattr(c, "k_bits"):
                layer_meta["k_bits"] = c.k_bits
                layer_meta["v_bits"] = c.v_bits
                layer_meta["k_group_size"] = c.k_group_size
                layer_meta["v_group_size"] = c.v_group_size
            with open(tmp_dir / f"layer_{i}_meta.json", "w") as f:
                json.dump(layer_meta, f)

        # Move to final location (replaces any existing entry)
        if entry_dir.exists():
            shutil.rmtree(entry_dir, ignore_errors=True)
        os.rename(str(tmp_dir), str(entry_dir))
    except Exception:
        # Clean up partial temp dir on any failure
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _load_cache_from_disk(cache_dir: Path, h: str) -> Optional[dict]:
    """Load a prompt cache entry from disk. Returns metadata + cache list."""
    entry_dir = cache_dir / h
    meta_path = entry_dir / "meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    prompt_cache = []
    for i in range(meta["n_layers"]):
        layer_meta_path = entry_dir / f"layer_{i}_meta.json"
        npz_path = entry_dir / f"layer_{i}.npz"

        if not layer_meta_path.exists():
            return None

        with open(layer_meta_path) as f:
            layer_meta = json.load(f)

        cls_name = layer_meta["class"]

        if cls_name == "MixedQuantKVCache":
            from .models.mixed_quant_cache import MixedQuantKVCache
            c = MixedQuantKVCache(
                k_bits=layer_meta["k_bits"],
                v_bits=layer_meta["v_bits"],
                k_group_size=layer_meta["k_group_size"],
                v_group_size=layer_meta["v_group_size"],
            )
            if npz_path.exists():
                arrays = dict(mx.load(str(npz_path)))
                n = len(arrays)
                if n >= 6:
                    c.keys = (arrays["s0"], arrays["s1"], arrays["s2"])
                    c.values = (arrays["s3"], arrays["s4"], arrays["s5"])
                    c.offset = layer_meta["offset"]
                else:
                    logger.warning(f"Incomplete cache layer {i}: {n}/6 arrays")
                    return None
            prompt_cache.append(c)
        else:
            # Standard KVCache
            c = KVCache()
            if npz_path.exists():
                arrays = dict(mx.load(str(npz_path)))
                if "s0" in arrays and "s1" in arrays:
                    c.update_and_fetch(arrays["s0"], arrays["s1"])
            prompt_cache.append(c)

    return {"meta": meta, "prompt_cache": prompt_cache}


def _delete_cache_from_disk(cache_dir: Path, model: Any, tokens: List[int]):
    """Remove a cache entry from disk."""
    h = _cache_key_hash(model, tokens)
    entry_dir = cache_dir / h
    if entry_dir.exists():
        shutil.rmtree(entry_dir, ignore_errors=True)


class DiskBackedPromptCache(LRUPromptCache):
    """LRU prompt cache that spills evicted entries to disk.

    When an entry is evicted from the in-memory LRU (size or byte
    limit exceeded), it is saved to `cache_dir`. When
    `fetch_nearest_cache` misses in RAM, the disk is checked.

    The disk index is built lazily from the directory listing on first
    fetch miss, so restarting the server with the same `cache_dir`
    automatically restores all previously-cached prompts.
    """

    def __init__(self, max_size: int = 10, cache_dir: str = "/tmp/mlx_kv_cache"):
        super().__init__(max_size=max_size)
        self._cache_dir = Path(cache_dir)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Cannot create prompt cache directory {cache_dir}: {e}. "
                "Disk persistence disabled."
            )
            self._cache_dir = None
        self._disk_index: Optional[dict] = None  # lazy-loaded
        if self._cache_dir:
            logger.info(
                f"Disk-backed prompt cache: {self._cache_dir} "
                "(not safe for multiple concurrent server instances)"
            )

    def _ensure_disk_index(self):
        """Scan cache_dir and build hash -> (model, tokens) index."""
        if self._disk_index is not None:
            return
        self._disk_index = {}
        if self._cache_dir is None or not self._cache_dir.exists():
            return
        # Clean up stale temp dirs from interrupted saves
        for tmp in self._cache_dir.glob(".tmp_*"):
            if tmp.is_dir():
                shutil.rmtree(tmp, ignore_errors=True)
                logger.info(f"Cleaned stale temp dir: {tmp.name}")

        for entry_dir in self._cache_dir.iterdir():
            if not entry_dir.is_dir() or entry_dir.name.startswith(".tmp_"):
                continue
            meta_path = entry_dir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self._disk_index[entry_dir.name] = {
                        "model": meta["model"],
                        "tokens": meta["tokens"],
                    }
                except Exception:
                    pass
        logger.info(f"Disk cache index: {len(self._disk_index)} entries")

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        # Track disk index size before insert (parent may evict)
        prev_size = len(self._lru)

        # Let the parent handle the LRU logic (may evict old entries)
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)

        # Persist the NEW entry to disk
        try:
            _save_cache_to_disk(
                self._cache_dir, model, tokens, prompt_cache, cache_type
            )
            h = _cache_key_hash(model, tokens)
            if self._disk_index is not None:
                self._disk_index[h] = {
                    "model": str(model),
                    "tokens": tokens,
                }
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

        # Parent's inline eviction (lines 1632-1639 of cache.py) doesn't
        # call our trim_to, so evicted entries become disk orphans. Cap
        # disk entries at 2x max_size to bound growth and prevent
        # reload-thrashing (fetching orphan from disk → re-insert →
        # immediately evicted again).
        if self._cache_dir is not None:
            self._cap_disk_size()

    def _cap_disk_size(self):
        """Remove oldest disk entries when exceeding 2x max_size."""
        if self._cache_dir is None:
            return
        entries = [
            d for d in self._cache_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        limit = self.max_size * 2
        if len(entries) <= limit:
            return
        # Sort by modification time, oldest first
        entries.sort(key=lambda d: d.stat().st_mtime)
        n_remove = len(entries) - limit
        for d in entries[:n_remove]:
            h = d.name
            shutil.rmtree(d, ignore_errors=True)
            if self._disk_index is not None and h in self._disk_index:
                del self._disk_index[h]
        logger.info(f"Capped disk cache: removed {n_remove} oldest entries")

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        # Try RAM first
        result, rest = super().fetch_nearest_cache(model, tokens)
        if result is not None:
            return result, rest

        # Cache miss in RAM — check disk
        self._ensure_disk_index()
        if not self._disk_index:
            return None, tokens

        # Look for exact match on disk
        h = _cache_key_hash(model, tokens)
        if h in self._disk_index:
            try:
                loaded = _load_cache_from_disk(self._cache_dir, h)
            except Exception as e:
                logger.warning(f"Corrupt disk cache entry {h}: {e}")
                loaded = None
            if loaded is not None:
                logger.info(
                    f"Disk cache hit: {len(loaded['meta']['tokens'])} tokens"
                )
                super().insert_cache(
                    model, tokens, loaded["prompt_cache"],
                    cache_type=loaded["meta"].get("cache_type", "assistant"),
                )
                return copy.deepcopy(loaded["prompt_cache"]), []

        # Look for longest prefix match on disk
        best_h = None
        best_len = 0
        for dh, info in self._disk_index.items():
            if str(info["model"]) != str(model):
                continue
            disk_tokens = info["tokens"]
            prefix_len = 0
            for a, b in zip(disk_tokens, tokens):
                if a != b:
                    break
                prefix_len += 1
            if prefix_len > best_len and prefix_len == len(disk_tokens):
                best_len = prefix_len
                best_h = dh

        if best_h is not None and best_len > 0:
            try:
                loaded = _load_cache_from_disk(self._cache_dir, best_h)
            except Exception as e:
                logger.warning(f"Corrupt disk cache entry {best_h}: {e}")
                loaded = None
            if loaded is not None:
                logger.info(
                    f"Disk cache prefix hit: {best_len}/{len(tokens)} tokens"
                )
                disk_tokens = loaded["meta"]["tokens"]
                super().insert_cache(
                    model, disk_tokens, loaded["prompt_cache"],
                    cache_type=loaded["meta"].get("cache_type", "assistant"),
                )
                return copy.deepcopy(loaded["prompt_cache"]), tokens[best_len:]

        return None, tokens

    def trim_to(self, *, n_sequences=None, n_bytes=None):
        """Trim LRU and also remove evicted entries from disk."""
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > n_sequences:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._delete_disk_entry(model, tokens)
        while self._n_bytes > n_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._delete_disk_entry(model, tokens)

    def _delete_disk_entry(self, model, tokens):
        """Remove a cache entry from disk and disk index."""
        if self._cache_dir is None:
            return
        h = _cache_key_hash(model, tokens)
        entry_dir = self._cache_dir / h
        if entry_dir.exists():
            shutil.rmtree(entry_dir, ignore_errors=True)
        if self._disk_index is not None and h in self._disk_index:
            del self._disk_index[h]
