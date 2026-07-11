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
  content-addressed (cid over model key + token ids) snapshot of a prefix's
  per-layer KV state, trimmed to exact length (no ``step`` slack).
* :class:`ForkedKVCache` — a per-layer cache mirroring :class:`KVCache`'s
  interface. It holds a *reference* to the frozen parent arrays (zero copy)
  plus a private, growing tail buffer. Writes go only to the tail, so the
  parent is structurally immutable: copy-on-write achieved by construction
  rather than by copy-on-first-write.
* :class:`PrefixForkRegistry` — a small cid-keyed registry with longest-prefix
  lookup (via ``PromptTrie``), LRU eviction, and content dedup.

THE MODEL-KEY CONTRACT
======================

Every registry call takes a ``model_key``: a caller-supplied, stable STRING
that identifies the *exact weights* the KV was computed with — it must encode
at least the model path, adapter, and revision (e.g.
``"org/model@rev+adapter-sha"``), and it MUST change whenever the weights
change (adapter load/unload, revision update, requantization). Cached KV is a
pure function of (weights, tokens); a key that outlives a weight change turns
into deterministic stale-KV false HITs. Non-string keys are rejected loudly:
an ``id()``-derived or object-identity key is forbidden because (a) an
in-place weight swap keeps the same object and (b) CPython recycles addresses
of dead objects, both of which alias distinct weight-identities onto one key.
As defense in depth, cid-dedup hits are spot-checked against the offered KV
content and raise on mismatch (see ``PrefixForkRegistry.freeze``).

ALIASING EXPERIMENT
===================

(mlx 0.32.0.dev, 2026-07.) The design question was whether a fork may share
the parent's *live* buffer by reference while the parent keeps decoding.
``KVCache.update_and_fetch`` writes via in-place slice assignment
(``keys[..., prev:offset, :] = new``) into a step(=256) preallocated buffer.
Empirically:

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
immutability is structural — no code path in this module ever calls
``__setitem__`` on the shared arrays, raw frozen array objects never escape
(every boundary hands out fresh zero-copy slice *objects*, so a consumer's
``__setitem__`` rebinds only the consumer's object), and the registry plus
each live fork hold references so donation is impossible by construction.

KNOWN COSTS AND LIMITS (v1)
===========================

* Each attention step sees ``concat(prefix, tail)``, a transient
  O(prefix + tail) buffer per layer per step (freed immediately by the memory
  pool). Attention already reads all KV bytes each step, so this adds one
  extra write pass; removing it needs a two-segment attention kernel and is
  left as a follow-up.
* Only stacks of plain ``KVCache`` layers are forkable; anything else makes
  ``freeze`` return ``None`` (a safe miss).
* A length-1 snapshot is only found by an exact-length fetch (``PromptTrie``'s
  ``shorter`` result requires a match past position 0) — a false miss, never
  a false hit.
* An exact-match ``fetch_fork`` returns ``remaining == []``; as with
  ``fetch_nearest_cache``, the caller must ensure at least one token is fed
  to the model (e.g. by keying snapshots on ``tokens[:-1]``).

Invalidation policy: evicting or invalidating a snapshot only removes its
*discoverability* — the next ``fetch_fork`` misses (a false MISS re-prefills,
which is always safe). Live forks keep the arrays alive through ordinary
Python references, so an eviction can never corrupt an in-flight generation
(a false HIT on freed memory is impossible by construction). Consequently
``max_bytes``/``nbytes`` bound only the *discoverable* set; snapshots pinned
by live forks are reported separately via ``pinned_nbytes``.
"""

import hashlib
import threading
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .models.cache import KVCache, PromptTrie, _BaseCache, create_attention_mask


def _require_model_key(model_key: Any) -> str:
    """Enforce the model-key contract (see module docstring): a stable str
    encoding model path + adapter + revision. Anything else raises."""
    if not isinstance(model_key, str):
        raise TypeError(
            "model_key must be a stable string identifying the exact weights "
            "(model path + adapter + revision, e.g. 'org/model@rev+adapter'); "
            f"got {type(model_key).__name__}. Object- or id()-derived keys "
            "are forbidden: in-place weight swaps keep the same object and "
            "CPython reuses addresses, both of which cause stale-KV false "
            "HITs. The key MUST change whenever the weights change."
        )
    return model_key


def compute_cid(model_key: str, tokens: List[int]) -> str:
    """Content-hash identity for a prefix: (model key, exact token ids).

    Same content -> same cid, so two independently frozen copies of the same
    prefix dedup to one snapshot. Distinct model keys never collide. The
    model key is subject to the module-level MODEL-KEY CONTRACT.
    """
    model_key = _require_model_key(model_key)
    h = hashlib.sha256()
    h.update(model_key.encode("utf-8", "replace"))
    for t in tokens:
        h.update(b"\x00" + str(int(t)).encode())
    return h.hexdigest()[:16]


class FrozenPrefixSnapshot:
    """An immutable, materialized, content-addressed prefix KV snapshot.

    ``keys[l]`` / ``values[l]`` are per-layer arrays of EXACT length
    ``len(tokens)`` (no step slack), copied out of the source cache with
    ``mx.array(...)`` and forced with ``mx.eval`` so they own their buffers —
    the source cache may keep decoding (or be garbage collected) without any
    effect on the snapshot. Consumers must never write to these arrays; the
    ForkedKVCache only ever reads them, and every boundary that could hand
    them out returns fresh slice objects instead (see module docstring).

    Construct via :meth:`freeze` (the blessed path, which materializes). The
    raw constructor stores fresh full-range slice *objects* of the arrays it
    is given, so a caller retaining (and later ``__setitem__``-ing) its own
    references cannot mutate the snapshot — but it does NOT copy: the caller
    must pass materialized arrays it will not share with a writer.
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
        # Fresh slice OBJECTS (zero-copy): a later __setitem__ through the
        # caller's original references rebinds the caller's objects, never
        # these pre-write nodes.
        self.keys = [k[:] for k in keys]
        self.values = [v[:] for v in values]
        self.nbytes = sum(k.nbytes + v.nbytes for k, v in zip(keys, values))
        # Live forks referencing this snapshot (weak: forks hold the strong
        # edge). Lets the registry report pinned-but-undiscoverable bytes.
        self._forks = weakref.WeakSet()

    def __len__(self):
        return len(self.tokens)

    @property
    def pinned(self) -> bool:
        """Whether any live fork still references this snapshot."""
        return len(self._forks) > 0

    @classmethod
    def freeze(
        cls, model_key: str, tokens: List[int], prompt_cache: List[Any]
    ) -> Optional["FrozenPrefixSnapshot"]:
        """Materialize a snapshot from a live prompt cache, or ``None``.

        Returns ``None`` (a safe miss) unless every layer is a plain
        ``KVCache`` whose offset matches ``len(tokens)`` — rotating, chunked,
        quantized, and stateful (Mamba) caches are not COW-forkable in v1.
        """
        model_key = _require_model_key(model_key)
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
        cid = compute_cid(model_key, tokens)
        return cls(cid, model_key, tuple(int(t) for t in tokens), keys, values)

    def fork(self) -> List["ForkedKVCache"]:
        """A fresh per-layer cache stack referencing this snapshot. O(1):
        no prefix bytes are copied or materialized."""
        forks = [
            ForkedKVCache(self.keys[l], self.values[l], snapshot=self)
            for l in range(len(self.keys))
        ]
        for f in forks:
            self._forks.add(f)
        return forks


class ForkedKVCache(_BaseCache):
    """A KVCache-compatible cache = frozen shared prefix + private tail.

    Reads (``update_and_fetch`` results, ``.state``) present the concatenation
    of the immutable prefix and the private tail; writes go exclusively into
    the tail's step-preallocated buffer (a plain :class:`KVCache`). The
    parent snapshot is therefore structurally immutable — no code path writes
    to the shared arrays — and any number of sibling forks may share it.

    Interface parity with :class:`KVCache`: ``update_and_fetch``, ``offset``,
    ``size``, ``state``, ``is_trimmable``/``trim``, ``make_mask``, ``empty``,
    ``nbytes``. Differences:

    * ``trim`` only reaches into the private tail — the frozen prefix cannot
      be trimmed (it is shared). ``trim(n)`` RAISES if asked to trim past the
      tail, because callers of ``trim_prompt_cache`` ignore its return value
      and would otherwise continue generating against skewed KV. Rollback of
      tokens generated *after* the fork (the intended use: speculative
      decoding, retries) always stays within the tail and works normally.
    * ``nbytes`` counts only the PRIVATE tail bytes: for byte budgeting, the
      shared prefix must be counted once (at the snapshot/registry), not once
      per fork. The shared portion is exposed as ``shared_nbytes``.
    * ``state`` is read-only. With an empty tail it returns fresh zero-copy
      slice objects of the frozen arrays (never the raw shared objects — a
      consumer ``__setitem__`` must not be able to corrupt every sibling
      fork); with a non-empty tail it materializes the joined arrays, an
      explicit O(prefix) copy. Use ``to_kv_cache()`` to detach into a plain,
      independent ``KVCache``.
    * ``meta_state`` raises: ``save_prompt_cache`` would otherwise write a
      file that cannot be loaded (``load_prompt_cache`` resolves classes from
      ``models/cache.py`` only). Detach with ``to_kv_cache()`` to serialize.
    * no ``to_quantized`` — ``maybe_quantize_kv_cache`` skips this cache via
      its ``hasattr`` guard (quantizing would have to copy the prefix anyway).
    """

    def __init__(
        self,
        prefix_keys: mx.array,
        prefix_values: mx.array,
        snapshot: Optional[FrozenPrefixSnapshot] = None,
    ):
        # Fresh slice objects: seal against later __setitem__ through the
        # caller's references (zero-copy; see module docstring).
        self._prefix_keys = prefix_keys[:]
        self._prefix_values = prefix_values[:]
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
            # Fresh zero-copy slice objects, NOT the stored references: a
            # consumer __setitem__ then rebinds only the consumer's object
            # and the frozen prefix (shared by every sibling fork) survives.
            return self._prefix_keys[:], self._prefix_values[:]
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

    @property
    def meta_state(self):
        raise ValueError(
            "ForkedKVCache cannot be serialized (save_prompt_cache would "
            "write a file load_prompt_cache cannot reconstruct); detach with "
            "to_kv_cache() first."
        )

    @meta_state.setter
    def meta_state(self, v):
        raise ValueError(
            "ForkedKVCache cannot be restored from serialized state; "
            "load a plain KVCache instead."
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
        # Only the private tail is trimmable; the shared frozen prefix is
        # not. Trimming past the tail must FAIL LOUDLY: trim_prompt_cache
        # callers ignore the returned count, so a silent clamp would leave
        # them generating against KV at the wrong offset.
        if n > self.tail.offset:
            raise ValueError(
                f"Cannot trim {n} tokens from a ForkedKVCache with a private "
                f"tail of {self.tail.offset}: the shared frozen prefix is "
                "immutable. Detach with to_kv_cache() to trim deeper."
            )
        return self.tail.trim(n)

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

    All methods take a ``model_key`` STRING subject to the module-level
    MODEL-KEY CONTRACT: it must identify the exact weights (model path +
    adapter + revision) and must change whenever the weights change. Non-str
    keys raise ``TypeError``.

    * ``freeze`` inserts (or dedups to) a snapshot for exact token content.
    * ``fetch_fork`` returns a zero-copy ``ForkedKVCache`` stack for the
      longest known snapshot that prefixes the requested tokens, plus the
      remaining suffix to prefill — the fork-aware analogue of
      ``LRUPromptCache.fetch_nearest_cache`` without its deepcopy.
    * Eviction (LRU by count and bytes) and ``invalidate`` only remove
      discoverability: misses are always safe (re-prefill), and live forks
      keep their snapshot's arrays alive via Python references, so a stale
      HIT on released memory cannot happen. ``max_bytes``/``nbytes`` bound
      the DISCOVERABLE set only; bytes still resident because live forks pin
      an evicted snapshot are reported by ``pinned_nbytes``.
    """

    def __init__(self, max_snapshots: int = 16, max_bytes: int = 1 << 63):
        self.max_snapshots = max_snapshots
        self.max_bytes = max_bytes
        self._snapshots: "OrderedDict[str, FrozenPrefixSnapshot]" = OrderedDict()
        self._trie = PromptTrie()  # namespaced by the model-key string
        # Weak refs to every snapshot ever inserted, for pinned_nbytes: a
        # snapshot evicted here stays alive exactly as long as forks pin it.
        self._tracked: Dict[str, "weakref.ref[FrozenPrefixSnapshot]"] = {}
        self._n_bytes = 0
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._snapshots)

    @property
    def nbytes(self):
        """Bytes of the DISCOVERABLE snapshots (what max_bytes bounds)."""
        return self._n_bytes

    @property
    def pinned_nbytes(self):
        """Bytes of snapshots kept resident by live forks — including ones
        already evicted/invalidated (discoverability and residency are
        deliberately decoupled; see class docstring).

        NOTE: a snapshot that is both discoverable and pinned is counted in
        BOTH ``nbytes`` and ``pinned_nbytes`` — the two are overlapping
        views (discoverable set vs resident set), not additive; do not sum
        them to estimate memory."""
        total = 0
        with self._lock:
            dead = []
            for cid, ref in self._tracked.items():
                snapshot = ref()
                if snapshot is None:
                    dead.append(cid)
                elif snapshot.pinned:
                    total += snapshot.nbytes
            for cid in dead:
                del self._tracked[cid]
        return total

    def freeze(
        self, model_key: str, tokens: List[int], prompt_cache: List[Any]
    ) -> Optional[str]:
        """Snapshot ``prompt_cache`` (which must cover exactly ``tokens``) and
        register it. Returns the snapshot cid, or ``None`` if the cache is not
        forkable (never raises for unsupported cache types — that is just a
        future miss).

        Dedup safety: if the cid already exists, the offered KV content is
        spot-checked against the stored snapshot and a mismatch RAISES —
        matching (model_key, tokens) with different KV means the key contract
        was violated (weights changed under a stale key)."""
        model_key = _require_model_key(model_key)
        cid = compute_cid(model_key, tokens)
        existing = self.get(cid)
        if existing is not None:
            self._dedup_spot_check(existing, tokens, prompt_cache)
            with self._lock:
                if cid in self._snapshots:
                    self._snapshots.move_to_end(cid)
            return cid
        snapshot = FrozenPrefixSnapshot.freeze(model_key, tokens, prompt_cache)
        if snapshot is None:
            return None
        with self._lock:
            existing = self._snapshots.get(snapshot.cid)
            if existing is not None:  # lost a freeze race: dedup
                self._dedup_spot_check(existing, tokens, prompt_cache)
                self._snapshots.move_to_end(snapshot.cid)
                return snapshot.cid
            self._snapshots[snapshot.cid] = snapshot
            self._tracked[snapshot.cid] = weakref.ref(snapshot)
            self._trie.add(snapshot.model_key, list(snapshot.tokens), snapshot.cid)
            self._n_bytes += snapshot.nbytes
            while len(self._snapshots) > self.max_snapshots or (
                self._n_bytes > self.max_bytes and len(self._snapshots) > 1
            ):
                self._evict_lru_locked()
        return snapshot.cid

    @staticmethod
    def _dedup_spot_check(
        snapshot: FrozenPrefixSnapshot, tokens: List[int], prompt_cache: List[Any]
    ):
        """Cheap content check on a cid-dedup hit: compare the last-4-token
        KV slice of EVERY layer (a layer-N-only weight change — e.g. a LoRA
        that leaves embeddings and early attention untouched — produces
        bit-identical early-layer KV, so a single-layer check is blind to
        it). Cannot prove equality (position window is limited), but
        catches per-layer-visible changes at freeze time and raises instead
        of silently serving stale KV. Freeze-time backstop only: fetch_fork
        under a stale key has nothing to compare against — the model-key
        contract is the primary defense."""
        if (
            len(prompt_cache) != len(snapshot.keys)
            or any(type(c) is not KVCache for c in prompt_cache)
            or prompt_cache[0].offset != len(tokens)
        ):
            return  # not comparable; a fresh freeze() would reject it too
        ok = True
        for layer, c in enumerate(prompt_cache):
            k_new = c.state[0]
            k_old = snapshot.keys[layer]
            if k_new.shape != k_old.shape or k_new.dtype != k_old.dtype:
                ok = False
                break
            w = min(4, k_old.shape[2])
            if not bool(mx.array_equal(k_old[..., -w:, :], k_new[..., -w:, :])):
                ok = False
                break
        if not ok:
            raise ValueError(
                f"prefix-fork dedup mismatch for cid {snapshot.cid}: same "
                "model_key and tokens but different KV content. The model "
                "key MUST change whenever the weights change (adapter load, "
                "revision update, requantization)."
            )

    def fetch_fork(
        self, model_key: str, tokens: List[int]
    ) -> Tuple[Optional[List[ForkedKVCache]], List[int]]:
        """Fork from the longest frozen snapshot prefixing ``tokens``.

        Returns ``(forks, remaining_tokens)`` where ``forks`` is a per-layer
        ``ForkedKVCache`` list (or ``None`` on a miss, with ``remaining ==
        tokens``). Unlike ``fetch_nearest_cache`` there is no trim-a-longer-
        cache path: a frozen prefix cannot be trimmed, so a longer-only match
        is deliberately a miss (false-MISS-only invariant).
        """
        model_key = _require_model_key(model_key)
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
