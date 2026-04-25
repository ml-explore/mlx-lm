"""Disk-backed LRU prompt cache for mlx_lm.server.

Wraps LRUPromptCache to persist evicted KV caches to disk and restore
them on cache miss. Survives server restarts.

Uses mlx-lm's own save_prompt_cache / load_prompt_cache for
serialization — handles all cache types (KVCache, CacheList,
QuantizedKVCache, etc.) correctly via safetensors.

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

from .models.cache import (
    LRUPromptCache,
    load_prompt_cache,
    save_prompt_cache,
)

logger = logging.getLogger(__name__)


def _cache_key_hash(model: Any, tokens: List[int]) -> str:
    """Stable hash for a (model, tokens) cache key."""
    raw = f"{model}:{','.join(str(t) for t in tokens)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _save_to_disk(cache_dir: Path, model: Any, tokens: List[int],
                   prompt_cache: List[Any], cache_type: str = "assistant"):
    """Save a prompt cache entry to disk atomically.

    Handles empty arrays (from uninitialized MoE sub-caches) by saving
    them separately in empty.json, since safetensors cannot serialize
    size-0 arrays.
    """
    if cache_dir is None:
        return
    h = _cache_key_hash(model, tokens)
    entry_dir = cache_dir / h
    tmp_dir = cache_dir / f".tmp_{h}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta = {
            "model": str(model),
            "tokens": tokens,
            "cache_type": cache_type,
        }
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Skip if any cache layer is uninitialized (keys=None).
        # KVCache.state crashes on empty caches.
        if any(c.empty() for c in prompt_cache):
            logger.warning("Skipping disk save: cache has uninitialized layers")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        # Collect cache state via mlx-lm's tree_flatten
        from mlx.utils import tree_flatten
        cache_data = [c.state for c in prompt_cache]
        cache_info = [c.meta_state for c in prompt_cache]
        cache_data_flat = dict(tree_flatten(cache_data))
        cache_classes = [type(c).__name__ for c in prompt_cache]

        # Extract empty arrays (safetensors can't serialize size=0)
        empty_arrays = {}
        for k, v in list(cache_data_flat.items()):
            if v.size == 0:
                empty_arrays[k] = {
                    "shape": [int(s) for s in v.shape],
                    "dtype": str(v.dtype).split(".")[-1],
                }
                del cache_data_flat[k]

        # Skip saving if no actual data (all empty or uninitialized)
        if not cache_data_flat and not empty_arrays:
            logger.warning(
                f"Skipping disk save: cache has no data "
                f"({len(prompt_cache)} layers, "
                f"{len(cache_classes)} classes: {set(cache_classes)})"
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        if empty_arrays:
            with open(tmp_dir / "empty.json", "w") as f:
                json.dump(empty_arrays, f)

        # Save via safetensors (now free of empty arrays)
        cache_metadata = [cache_info, {}, cache_classes]
        cache_metadata_flat = dict(tree_flatten(cache_metadata))
        import mlx.core as mx
        mx.save_safetensors(
            str(tmp_dir / "cache.safetensors"),
            cache_data_flat,
            cache_metadata_flat,
        )

        if entry_dir.exists():
            shutil.rmtree(entry_dir, ignore_errors=True)
        os.rename(str(tmp_dir), str(entry_dir))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _load_from_disk(cache_dir: Path, h: str) -> Optional[dict]:
    """Load a prompt cache entry from disk.

    Re-inserts empty arrays from empty.json before reconstructing
    the cache (reverses the save-side workaround for safetensors).
    """
    entry_dir = cache_dir / h
    meta_path = entry_dir / "meta.json"
    cache_path = entry_dir / "cache.safetensors"

    if not meta_path.exists() or not cache_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Load arrays + metadata from safetensors
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    arrays, cache_metadata = mx.load(str(cache_path), return_metadata=True)

    # Re-insert empty arrays saved separately
    empty_path = entry_dir / "empty.json"
    if empty_path.exists():
        with open(empty_path) as f:
            empty_arrays = json.load(f)
        for k, info in empty_arrays.items():
            dtype = getattr(mx, info["dtype"], mx.float32)
            arrays[k] = mx.zeros(info["shape"], dtype=dtype)

    arrays = tree_unflatten(list(arrays.items()))
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    if not cache_metadata or len(cache_metadata) < 3:
        raise ValueError(
            f"Corrupt cache metadata: expected 3 elements, got {len(cache_metadata) if cache_metadata else 0}"
        )
    info, metadata, classes = cache_metadata

    # Inject cache classes into cache.py's globals so CacheList.from_state
    # can resolve sub-cache types (it uses its own module globals)
    # Ensure custom cache classes are in cache.py's globals so
    # CacheList.from_state can resolve sub-cache types. Injected
    # unconditionally because sub-cache class names are buried
    # inside CacheList.meta_state, not in the top-level classes list.
    import mlx_lm.models.cache as _cache_mod
    try:
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        _cache_mod.__dict__.setdefault("TurboQuantKVCache", TurboQuantKVCache)
    except ImportError:
        pass
    try:
        from mlx_lm.models.mixed_quant_cache import MixedQuantKVCache
        _cache_mod.__dict__.setdefault("MixedQuantKVCache", MixedQuantKVCache)
    except ImportError:
        pass

    local_globals = _cache_mod.__dict__

    # Allowlist: only permit known cache classes from cache.py to prevent
    # arbitrary class instantiation from crafted safetensors files.
    _ALLOWED_CACHE_CLASSES = {
        "KVCache", "QuantizedKVCache", "RotatingKVCache",
        "CacheList", "BatchKVCache", "BatchRotatingKVCache",
        "ConcatenateKVCache", "ArraysCache", "ChunkedKVCache",
        "TurboQuantKVCache", "MixedQuantKVCache",
    }
    for c in classes:
        if c not in _ALLOWED_CACHE_CLASSES:
            raise ValueError(
                f"Untrusted cache class '{c}' in disk cache. "
                f"Allowed: {_ALLOWED_CACHE_CLASSES}"
            )

    prompt_cache = [
        local_globals[c].from_state(state, meta_state)
        for c, state, meta_state in zip(classes, arrays, info)
    ]
    return {"meta": meta, "prompt_cache": prompt_cache}


class DiskBackedPromptCache(LRUPromptCache):
    """LRU prompt cache that persists entries to disk.

    On insert: saves to disk (for restart survival).
    On cache miss in RAM: checks disk before giving up.
    Disk entries capped at 2x max_size by mtime.
    """

    def __init__(self, max_size: int = 10, cache_dir: str = "~/.cache/mlx_kv_cache"):
        super().__init__(max_size=max_size)
        self._cache_dir = Path(cache_dir).expanduser()
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Cannot create prompt cache directory {cache_dir}: {e}. "
                "Disk persistence disabled."
            )
            self._cache_dir = None
        self._disk_index: Optional[dict] = None
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
            cache_path = entry_dir / "cache.safetensors"
            if meta_path.exists() and cache_path.exists():
                # Reject empty safetensors (header-only, no data)
                if cache_path.stat().st_size < 64:
                    shutil.rmtree(entry_dir, ignore_errors=True)
                    logger.info(f"Removed empty cache entry: {entry_dir.name}")
                    continue
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self._disk_index[entry_dir.name] = {
                        "model": meta["model"],
                        "tokens": meta["tokens"],
                    }
                except Exception:
                    shutil.rmtree(entry_dir, ignore_errors=True)
            elif entry_dir.is_dir():
                # Old format or incomplete entry — clean up
                shutil.rmtree(entry_dir, ignore_errors=True)
                logger.info(f"Removed incompatible cache entry: {entry_dir.name}")
        logger.info(f"Disk cache index: {len(self._disk_index)} entries")

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        # Track LRU size before insert (parent may evict)
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)

        # Persist to disk
        try:
            _save_to_disk(
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

        # Cap disk entries to prevent unbounded growth
        if self._cache_dir is not None:
            self._cap_disk_size()

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        # Try RAM first
        result, rest = super().fetch_nearest_cache(model, tokens)
        if result is not None:
            return result, rest

        # Cache miss in RAM — check disk
        self._ensure_disk_index()
        if not self._disk_index:
            return None, tokens

        # Exact match on disk
        h = _cache_key_hash(model, tokens)
        if h in self._disk_index:
            try:
                loaded = _load_from_disk(self._cache_dir, h)
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

        # Longest prefix match on disk
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
                loaded = _load_from_disk(self._cache_dir, best_h)
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
        """Trim LRU and remove evicted entries from disk.

        Delegates to parent's trim logic and tracks which entries get
        evicted so we can also clean them from disk. This avoids
        duplicating the parent's eviction algorithm.
        """
        evicted = []
        original_pop = self._lru.pop

        def tracking_pop():
            result = original_pop()
            evicted.append(result)
            return result

        self._lru.pop = tracking_pop
        try:
            super().trim_to(n_sequences=n_sequences, n_bytes=n_bytes)
        finally:
            self._lru.pop = original_pop

        for model, tokens in evicted:
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
        entries.sort(key=lambda d: d.stat().st_mtime)
        n_remove = len(entries) - limit
        for d in entries[:n_remove]:
            h = d.name
            shutil.rmtree(d, ignore_errors=True)
            if self._disk_index is not None and h in self._disk_index:
                del self._disk_index[h]
        logger.info(f"Capped disk cache: removed {n_remove} oldest entries")
