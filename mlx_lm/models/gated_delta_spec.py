# Copyright © 2026 Apple Inc.

"""GatedDeltaNet rollback support for MTP speculative decoding.

Speculative decoding at draft depth 1 verifies the sequence ``[t_n, d_1]``
in a single main-model forward. On rejection, the cache must be rolled
back to the "after t_n" state so the next iteration sees clean state.

``KVCache`` is naturally trimmable (just decrement ``offset``), but the
``ArraysCache`` used by ``GatedDeltaNet`` holds a recurrent state that was
advanced through both tokens -- there is no in-place inverse.

This module installs an opt-in patch on
``mlx_lm.models.qwen3_5.GatedDeltaNet`` (and
``mlx_lm.models.qwen3_next.Qwen3NextGatedDeltaNet``): when a cache is
marked with the ``_spec_capture`` flag and the input has ``S == 2``, the
patched ``__call__`` runs the GDN block twice with ``S == 1`` each,
snapshotting the intermediate cache state between the two calls. The
output is bit-exact with the ``S == 2`` path because ``conv1d`` and
``gated_delta_update`` both evolve sequentially anyway.

Cost: 2x kernel launches per GDN layer per verify cycle. Trades a full
rejection re-forward (40 layers worth) for ~30 GDN layers x one extra
launch each, which is net-positive on Apple Silicon.

When ``_spec_capture`` is unset (or the input has ``S != 2``), the patch
is a no-op and the original GDN code path runs unchanged.

Public entry points:
* :func:`install` -- idempotently install the patch.
* :func:`mark_for_capture` / :func:`unmark_capture` -- toggle capture
  mode on a cache list.
* :func:`rollback_to_intermediate` -- on rejection, roll
  ``ArraysCache`` entries back to the captured intermediate and trim each
  ``KVCache`` by one position.
"""

from __future__ import annotations

from typing import Any, List, Optional


_PATCHED_FLAG = "_mlx_lm_spec_patched"

# Per-class state: original ``__call__`` is held in ``_originals[cls]`` so
# the patched dispatch can delegate to it. We patch the bound type method
# rather than per-instance, since instance ``__call__`` patches are not
# fired through Python's descriptor protocol.
_originals: dict = {}


def _snap_cache_refs(cache):
    """Snapshot the GDN cache slots by reference (no tensor copy).

    ``GatedDeltaNet`` reassigns ``cache[0]``/``cache[1]`` rather than
    mutating in place, and ``advance()`` reassigns ``lengths`` /
    ``left_padding`` via ``-=`` (which produces new arrays), so a
    by-reference snapshot is safe.
    """
    return (
        list(cache.cache),
        cache.lengths,
        cache.left_padding,
    )


def _restore_cache_refs(cache, snap):
    cache.cache = list(snap[0])
    cache.lengths = snap[1]
    cache.left_padding = snap[2]


def _make_patched_call(cls):
    """Build the patched ``__call__`` for a specific GDN class."""
    original = _originals[cls]

    def _patched(self, inputs, mask=None, cache=None):
        if (
            cache is not None
            and getattr(cache, "_spec_capture", False)
            and inputs.shape[1] == 2
        ):
            # Split mode: process the two tokens one at a time, capturing
            # the cache state between them.
            half0 = inputs[:, :1, :]
            half1 = inputs[:, 1:, :]
            mask0 = None if mask is None else mask[..., :1]
            mask1 = None if mask is None else mask[..., 1:]

            out0 = original(self, half0, mask0, cache)
            # Snapshot cache state after the first token.
            cache._spec_intermediate = _snap_cache_refs(cache)
            out1 = original(self, half1, mask1, cache)
            import mlx.core as mx  # local import: cheap, avoids hard dep at module load

            return mx.concatenate([out0, out1], axis=1)
        return original(self, inputs, mask, cache)

    return _patched


def install() -> None:
    """Idempotently install the spec-decode capture patch.

    Patches:
        * ``mlx_lm.models.qwen3_5.GatedDeltaNet``
        * ``mlx_lm.models.qwen3_next.Qwen3NextGatedDeltaNet``
    """
    from .qwen3_5 import GatedDeltaNet as Qwen35GDN
    from .qwen3_next import Qwen3NextGatedDeltaNet

    for cls in (Qwen35GDN, Qwen3NextGatedDeltaNet):
        if getattr(cls, _PATCHED_FLAG, False):
            continue
        _originals[cls] = cls.__call__
        cls.__call__ = _make_patched_call(cls)
        setattr(cls, _PATCHED_FLAG, True)


def _is_arrays_cache(c) -> bool:
    return hasattr(c, "cache") and isinstance(c.cache, list)


def _is_kv_cache(c) -> bool:
    return hasattr(c, "offset") and hasattr(c, "keys") and not _is_arrays_cache(c)


def mark_for_capture(cache_list: List[Any]) -> None:
    """Mark every ``ArraysCache`` in ``cache_list`` to capture intermediate
    state on the next 2-token forward."""
    for c in cache_list:
        if _is_arrays_cache(c):
            c._spec_capture = True
            c._spec_intermediate = None


def unmark_capture(cache_list: List[Any]) -> None:
    """Disable capture mode after a verify forward."""
    for c in cache_list:
        if _is_arrays_cache(c):
            c._spec_capture = False


def rollback_to_intermediate(cache_list: List[Any]) -> bool:
    """Roll all caches back to the post-first-token state from the most
    recent verify forward.

    For ``ArraysCache`` entries: restore the snapshot taken during the
    split-call. For ``KVCache``-style entries: trim ``offset`` by one to
    drop the draft's K/V row.

    Returns ``True`` if every relevant ``ArraysCache`` had a captured
    snapshot. Returns ``False`` if any ``ArraysCache`` was missing one
    (should not happen if ``mark_for_capture`` was paired with the verify
    forward).
    """
    ok = True
    for c in cache_list:
        if _is_arrays_cache(c):
            snap = getattr(c, "_spec_intermediate", None)
            if snap is None:
                ok = False
                continue
            _restore_cache_refs(c, snap)
            c._spec_intermediate = None
        elif hasattr(c, "offset"):  # KVCache / ChunkedKVCache / RotatingKVCache
            c.offset = max(0, c.offset - 1)
    return ok


# --- Multi-token verify support --------------------------------------------
#
# The mark_for_capture / rollback_to_intermediate path above is depth=1
# specific: it triggers only when the verify forward has S == 2 and
# captures ONE intermediate (post-t_n, pre-d_1).
#
# For depth-k verify (S = k + 1), partial accept could need to roll back
# anywhere in [0, k - 1]. The simplest correct approach: snapshot the
# WHOLE pre-verify state and, on partial accept, restore + re-forward the
# accepted prefix. One extra forward on partial accept; the GDN math
# stays exact (we never invert the recurrence).


def _arrays_cache_snap(cache):
    return (
        "arrays",
        list(cache.cache),
        cache.lengths,
        cache.left_padding,
    )


def _kv_cache_snap(cache):
    return ("kv", cache.offset)


def _restore_snap(cache, snap):
    kind = snap[0]
    if kind == "kv":
        cache.offset = snap[1]
    elif kind == "arrays":
        cache.cache = list(snap[1])
        cache.lengths = snap[2]
        cache.left_padding = snap[3]
    else:
        raise ValueError(f"unknown snap kind {kind!r}")


def snapshot_pre_verify(cache_list: List[Any]) -> List[Optional[tuple]]:
    """Snapshot the pre-verify state of every cache.

    Call this BEFORE a multi-token verify forward. On partial accept,
    pass the result to :func:`restore_pre_verify` to revert all caches,
    then re-forward the accepted prefix to advance them.
    """
    snap: List[Optional[tuple]] = []
    for c in cache_list:
        if _is_arrays_cache(c):
            snap.append(_arrays_cache_snap(c))
        elif hasattr(c, "offset"):
            snap.append(_kv_cache_snap(c))
        else:
            snap.append(None)
    return snap


def restore_pre_verify(cache_list: List[Any], snap: List[Optional[tuple]]) -> None:
    """Restore caches to a snapshot returned by :func:`snapshot_pre_verify`."""
    for c, s in zip(cache_list, snap):
        if s is None:
            continue
        _restore_snap(c, s)
