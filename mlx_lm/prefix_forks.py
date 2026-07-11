# Copyright © 2026 Apple Inc.

"""Copy-on-write prefix forks for KV caches.

Many concurrent requests often share one long common prefix (a system prompt,
a document, a few-shot preamble). The existing request-level prompt cache
(``LRUPromptCache``) hands every hit a ``copy.deepcopy`` of the cached KV
stack — the ``deepcopy`` call itself is cheap (lazy arrays), but the copy is
materialized on first use, costing O(prefix) memory *and* wall-clock per
request (multiple seconds at long contexts).

This module lets N requests fork from ONE frozen, in-RAM parent snapshot,
paying only for their private suffixes:

* :class:`FrozenPrefixSnapshot` — an immutable, ``mx.eval``-materialized,
  content-addressed (cid over token ids) snapshot of a prefix's per-layer KV
  state, trimmed to exact length (no ``step`` slack).
* :class:`ForkedKVCache` — a per-layer cache mirroring :class:`KVCache`'s
  interface. It holds a *reference* to the frozen parent arrays (zero copy)
  plus a private, growing tail buffer. Writes go only to the tail, so the
  parent is structurally immutable: copy-on-write achieved by construction
  rather than by copy-on-first-write.
* :class:`PrefixForkRegistry` — a small cid-keyed registry with longest-prefix
  lookup (via ``PromptTrie``), LRU eviction, and content dedup.

Aliasing experiment (mlx 0.32.0.dev, 2026-07): the design question was whether
a fork may share the parent's *live* buffer by reference while the parent
keeps decoding. ``KVCache.update_and_fetch`` writes via in-place slice
assignment (``keys[..., prev:offset, :] = new``) into a step(=256)
preallocated buffer. Empirically:

* Mutation visibility follows PYTHON-OBJECT identity, not buffer identity.
  ``a[...] = x`` (``__setitem__``) rebinds the array *object* to the
  ``slice_update`` result, so every reference through that same Python object
  observes the write — but any slice taken *before* the write (lazy or
  evaluated) references the pre-write graph node and is preserved. MLX's
  donation machinery never recycles a buffer that another live node
  references, so even the refcount-hungry decode loop cannot scribble an
  exported slice.
* HOWEVER, an (evaluated) slice of the step-padded ``(B, H, S, D)`` buffer
  along axis 2 is strided, so materializing it costs an O(prefix) copy — per
  fork. And a lazy slice pins the parent's entire step-padded buffer alive
  while remaining one MLX donation-optimization away from unsoundness (live
  ``.state`` aliases have scribbled snapshots before; see the ``mx.array`` +
  ``mx.eval`` park recipe this module reuses).

Conclusion: materialize ONCE per unique prefix content into a frozen,
exact-length snapshot, and let every fork share those arrays by reference.
That single O(prefix) copy is amortized across all forks (content-addressed
dedup), fork creation is O(1), a fork's persistent footprint is O(tail), and
immutability is structural — nothing ever calls ``__setitem__`` on the shared
arrays, and the registry (plus each live fork) holds references so donation
is impossible by construction.

Known v1 cost: each attention step sees ``concat(prefix, tail)``, a transient
O(prefix + tail) buffer per layer per step (freed immediately by the memory
pool). Attention already reads all KV bytes each step, so this adds one extra
write pass; removing it needs a two-segment attention kernel and is left as a
follow-up.

Invalidation policy: evicting or invalidating a snapshot only removes its
*discoverability* — the next ``fetch_fork`` misses (a false MISS re-prefills,
which is always safe). Live forks keep the arrays alive through ordinary
Python references, so an eviction can never corrupt an in-flight generation
(a false HIT on freed memory is impossible by construction).
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import mlx.core as mx

from .models.cache import KVCache, PromptTrie, _BaseCache, create_attention_mask


def compute_cid(model_key: str, tokens: List[int]) -> str:
    """Content-hash identity for a prefix: (model namespace, exact token ids).

    Same content -> same cid, so two independently frozen copies of the same
    prefix dedup to one snapshot. Different models never collide.
    """
    h = hashlib.sha256()
    h.update(model_key.encode("utf-8", "replace"))
    for t in tokens:
        h.update(b"\x00" + str(int(t)).encode())
    return h.hexdigest()[:16]


def _model_key(model: Any) -> str:
    """A stable-within-process namespace string for a model object."""
    if isinstance(model, str):
        return model
    return f"id:{id(model)}"


class FrozenPrefixSnapshot:
    """An immutable, materialized, content-addressed prefix KV snapshot.

    ``keys[l]`` / ``values[l]`` are per-layer arrays of EXACT length
    ``len(tokens)`` (no step slack), copied out of the source cache with
    ``mx.array(...)`` and forced with ``mx.eval`` so they own their buffers —
    the source cache may keep decoding (or be garbage collected) without any
    effect on the snapshot. Consumers must never write to these arrays; the
    ForkedKVCache only ever reads them.
    """

    def __init__(
        self,
        cid: str,
        model_key: str,
        tokens: Tuple[int, ...],
        keys: List[mx.array],
        values: List[mx.array],
    ):
        self.cid = cid
        self.model_key = model_key
        self.tokens = tokens
        self.keys = keys
        self.values = values
        self.nbytes = sum(k.nbytes + v.nbytes for k, v in zip(keys, values))

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def freeze(
        cls, model: Any, tokens: List[int], prompt_cache: List[Any]
    ) -> Optional["FrozenPrefixSnapshot"]:
        """Materialize a snapshot from a live prompt cache, or ``None``.

        Returns ``None`` (a safe miss) unless every layer is a plain
        ``KVCache`` whose offset matches ``len(tokens)`` — rotating, chunked,
        quantized, and stateful (Mamba) caches are not COW-forkable in v1.
        """
        if len(tokens) == 0 or len(prompt_cache) == 0:
            return None
        for c in prompt_cache:
            if type(c) is not KVCache or c.offset != len(tokens):
                return None
        keys, values = [], []
        for c in prompt_cache:
            # .state may be a lazy slice of (or at full capacity, the very
            # same Python object as) the live buffer the source cache keeps
            # writing into. mx.array + mx.eval takes an independent,
            # exact-length device buffer NOW (the proven park recipe).
            k, v = c.state
            keys.append(mx.array(k))
            values.append(mx.array(v))
        mx.eval(*keys, *values)
        cid = compute_cid(_model_key(model), tokens)
        return cls(cid, _model_key(model), tuple(int(t) for t in tokens), keys, values)

    def fork(self) -> List["ForkedKVCache"]:
        """A fresh per-layer cache stack referencing this snapshot. O(1):
        no prefix bytes are copied or materialized."""
        return [
            ForkedKVCache(self.keys[l], self.values[l], snapshot=self)
            for l in range(len(self.keys))
        ]


class ForkedKVCache(_BaseCache):
    """A KVCache-compatible cache = frozen shared prefix + private tail.

    Reads (``update_and_fetch`` results, ``.state``) present the concatenation
    of the immutable prefix and the private tail; writes go exclusively into
    the tail's step-preallocated buffer (a plain :class:`KVCache`). The
    parent snapshot is therefore structurally immutable — no code path writes
    to the shared arrays — and any number of sibling forks may share it.

    Interface parity with :class:`KVCache`: ``update_and_fetch``, ``offset``,
    ``size``, ``state``, ``meta_state``, ``is_trimmable``/``trim``,
    ``make_mask``, ``empty``, ``nbytes``. Differences:

    * ``trim`` only reaches into the private tail — the frozen prefix cannot
      be trimmed (it is shared). ``trim(n)`` returns the number actually
      trimmed, clamped to the tail length.
    * ``nbytes`` counts only the PRIVATE tail bytes: for byte budgeting, the
      shared prefix must be counted once (at the snapshot/registry), not once
      per fork. The shared portion is exposed as ``shared_nbytes``.
    * ``state`` materializes the joined (prefix + tail) arrays — an explicit
      O(prefix) copy for serialization; use ``to_kv_cache()`` to detach into
      a plain, independent ``KVCache``. ``state`` is read-only.
    * no ``to_quantized`` — ``maybe_quantize_kv_cache`` skips this cache via
      its ``hasattr`` guard (quantizing would have to copy the prefix anyway).
    """

    step = 256

    def __init__(
        self,
        prefix_keys: mx.array,
        prefix_values: mx.array,
        snapshot: Optional[FrozenPrefixSnapshot] = None,
    ):
        self._prefix_keys = prefix_keys
        self._prefix_values = prefix_values
        self._prefix_len = prefix_keys.shape[2]
        # Keep the snapshot alive: registry eviction only removes
        # discoverability, never the arrays under a live fork.
        self._snapshot = snapshot
        self.tail = KVCache()

    @property
    def offset(self):
        return self._prefix_len + self.tail.offset

    @property
    def prefix_length(self):
        return self._prefix_len

    def update_and_fetch(self, keys, values):
        tail_keys, tail_values = self.tail.update_and_fetch(keys, values)
        # The concat reads the frozen prefix (never writes it) and produces a
        # fresh transient buffer for attention; the tail slice stays a view of
        # the private buffer that the next step writes in place.
        return (
            mx.concatenate([self._prefix_keys, tail_keys], axis=2),
            mx.concatenate([self._prefix_values, tail_values], axis=2),
        )

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.tail.offset == 0:
            return self._prefix_keys, self._prefix_values
        tail_keys, tail_values = self.tail.state
        return (
            mx.concatenate([self._prefix_keys, tail_keys], axis=2),
            mx.concatenate([self._prefix_values, tail_values], axis=2),
        )

    @state.setter
    def state(self, v):
        raise ValueError(
            "ForkedKVCache state cannot be set; detach with to_kv_cache() first."
        )

    def to_kv_cache(self) -> KVCache:
        """Detach into an independent plain KVCache (explicit O(prefix) copy)."""
        cache = KVCache()
        keys, values = self.state
        keys, values = mx.array(keys), mx.array(values)
        mx.eval(keys, values)
        cache.state = (keys, values)
        return cache

    def is_trimmable(self):
        return True

    def trim(self, n):
        # Only the private tail is trimmable; the shared prefix is frozen.
        return self.tail.trim(min(n, self.tail.offset))

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return False

    @property
    def nbytes(self):
        """Private (per-fork) bytes only; see class docstring."""
        return self.tail.nbytes

    @property
    def shared_nbytes(self):
        return self._prefix_keys.nbytes + self._prefix_values.nbytes


class PrefixForkRegistry:
    """cid-keyed store of frozen prefix snapshots with longest-prefix fetch.

    * ``freeze`` inserts (or dedups to) a snapshot for exact token content.
    * ``fetch_fork`` returns a zero-copy ``ForkedKVCache`` stack for the
      longest known snapshot that prefixes the requested tokens, plus the
      remaining suffix to prefill — the fork-aware analogue of
      ``LRUPromptCache.fetch_nearest_cache`` without its deepcopy.
    * Eviction (LRU by count and bytes) and ``invalidate`` only remove
      discoverability: misses are always safe (re-prefill), and live forks
      keep their snapshot's arrays alive via Python references, so a stale
      HIT on released memory cannot happen.
    """

    def __init__(self, max_snapshots: int = 16, max_bytes: int = 1 << 63):
        self.max_snapshots = max_snapshots
        self.max_bytes = max_bytes
        self._snapshots: "OrderedDict[str, FrozenPrefixSnapshot]" = OrderedDict()
        # The trie is namespaced by the model-key STRING (nn.Module is
        # dict-derived and unhashable, so the object itself cannot key it).
        self._trie = PromptTrie()
        self._n_bytes = 0
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._snapshots)

    @property
    def nbytes(self):
        return self._n_bytes

    def freeze(
        self, model: Any, tokens: List[int], prompt_cache: List[Any]
    ) -> Optional[str]:
        """Snapshot ``prompt_cache`` (which must cover exactly ``tokens``) and
        register it. Returns the snapshot cid, or ``None`` if the cache is not
        forkable (never raises for unsupported cache types — that is just a
        future miss)."""
        cid = compute_cid(_model_key(model), tokens)
        with self._lock:
            if cid in self._snapshots:
                self._snapshots.move_to_end(cid)  # dedup: same content, one copy
                return cid
        snapshot = FrozenPrefixSnapshot.freeze(model, tokens, prompt_cache)
        if snapshot is None:
            return None
        with self._lock:
            if snapshot.cid in self._snapshots:  # lost a freeze race: dedup
                self._snapshots.move_to_end(snapshot.cid)
                return snapshot.cid
            self._snapshots[snapshot.cid] = snapshot
            self._trie.add(snapshot.model_key, list(snapshot.tokens), snapshot.cid)
            self._n_bytes += snapshot.nbytes
            while len(self._snapshots) > self.max_snapshots or (
                self._n_bytes > self.max_bytes and len(self._snapshots) > 1
            ):
                self._evict_lru_locked()
        return snapshot.cid

    def fetch_fork(
        self, model: Any, tokens: List[int]
    ) -> Tuple[Optional[List[ForkedKVCache]], List[int]]:
        """Fork from the longest frozen snapshot prefixing ``tokens``.

        Returns ``(forks, remaining_tokens)`` where ``forks`` is a per-layer
        ``ForkedKVCache`` list (or ``None`` on a miss, with ``remaining ==
        tokens``). Unlike ``fetch_nearest_cache`` there is no trim-a-longer-
        cache path: a frozen prefix cannot be trimmed, so a longer-only match
        is deliberately a miss (false-MISS-only invariant).
        """
        model_key = _model_key(model)
        with self._lock:
            result = self._trie.search(model_key, tokens)
            matched = result.exact if result.exact is not None else result.shorter
            if matched is None:
                return None, tokens
            cid = self._trie.get(model_key, matched)
            snapshot = self._snapshots.get(cid)
            if snapshot is None:  # stale trie entry: treat as a miss
                return None, tokens
            self._snapshots.move_to_end(cid)
        return snapshot.fork(), tokens[len(matched) :]

    def get(self, cid: str) -> Optional[FrozenPrefixSnapshot]:
        with self._lock:
            return self._snapshots.get(cid)

    def invalidate(self, cid: str) -> bool:
        """Remove a snapshot from the registry (future fetches miss). Live
        forks referencing it are unaffected. Returns whether it was held."""
        with self._lock:
            snapshot = self._snapshots.pop(cid, None)
            if snapshot is None:
                return False
            self._remove_locked(snapshot)
            return True

    def _remove_locked(self, snapshot: FrozenPrefixSnapshot):
        self._n_bytes -= snapshot.nbytes
        try:
            self._trie.pop(snapshot.model_key, list(snapshot.tokens))
        except KeyError:
            pass  # already unlinked; the miss it causes is safe

    def _evict_lru_locked(self):
        cid, snapshot = self._snapshots.popitem(last=False)
        self._remove_locked(snapshot)
