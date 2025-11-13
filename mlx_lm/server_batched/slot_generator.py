# ABOUTME: Coordinates slot-indexed prompt prefill and decode loops.
# ABOUTME: Provides vectorized model calls while managing slot assignments.

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from ..models.base import use_attention_backend
from ..models.cache import PagedKVCacheAdapter, make_prompt_cache
from .array_decode_runner import ArrayDecodeRunner
from .paged_arrays import ArrayBatchView, PagedArraysProvider
from .paged_context import batch_view_scope
from .prefill_array_runner import PrefillArrayRunner
from .safe_eval import safe_eval

_BENCH_TRACE = os.environ.get("MLXLM_BENCH_TRACE", "").lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}
_BENCH_LOG = logging.getLogger("mlx_lm.bench")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "off", "no"}


_PAGED_TRACE = _env_flag("MLXLM_PAGED_TRACE")
_PAGED_LOG = logging.getLogger("mlx_lm.paged")
_COMPILE_DECODE = _env_flag("MLXLM_COMPILE_DECODE")


def _array_preview(arr, limit: int = 6):
    if arr is None:
        return None
    try:
        safe_eval(arr)
        data = arr.tolist()
    except Exception as exc:  # pragma: no cover - debug helper
        return f"<err:{exc}>"
    if isinstance(data, list) and len(data) > limit:
        preview = list(data[:limit])
        preview.append("...")
        return preview
    return data


def _seq_preview(values: Sequence[int], limit: int = 6):
    seq = list(values)
    if len(seq) > limit:
        trimmed = list(seq[:limit])
        trimmed.append("...")
        return trimmed
    return seq


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _mx_array(data, *, dtype=None):
    if dtype is None and hasattr(mx, "int32"):
        dtype = mx.int32
    try:
        if dtype is not None:
            return mx.array(data, dtype=dtype)
        return mx.array(data)
    except TypeError:
        return mx.array(data)


class PrefillSlicer:
    def __init__(
        self,
        *,
        enabled: bool,
        target_ms: float,
        hard_cap_ms: float,
        min_tokens: int,
        max_tokens: int,
        alpha: float = 0.25,
    ) -> None:
        self.enabled = enabled
        self.target_ms = max(1.0, target_ms)
        self.hard_cap_ms = max(self.target_ms, hard_cap_ms)
        self.min_tokens = max(1, min_tokens)
        self.max_tokens = max(self.min_tokens, max_tokens)
        self.alpha = max(0.01, min(alpha, 1.0))
        self.ewma_tok_ms = self.target_ms / max(self.min_tokens, 1)
        self.last_slice_ms = 0.0
        self.last_slice_tokens = 0
        self.idle_streak = 0

    def next_length(self, remaining: int) -> int:
        if not self.enabled or remaining <= self.min_tokens:
            return remaining
        base = max(self.ewma_tok_ms, 0.05)
        suggestion = int(math.ceil(self.target_ms / base))
        suggestion = max(self.min_tokens, min(suggestion, self.max_tokens))
        if self.last_slice_ms and self.last_slice_ms < self.target_ms * 0.5:
            scale = 1.0 + min(self.idle_streak, 5) * 0.1
            suggestion = min(int(suggestion * scale), self.max_tokens)
        if self.last_slice_ms and self.last_slice_ms > self.hard_cap_ms:
            suggestion = max(self.min_tokens, suggestion // 2)
        return max(1, min(suggestion, remaining))

    def observe(self, tokens: int, wall_ms: float) -> None:
        if tokens <= 0:
            return
        self.last_slice_ms = wall_ms
        self.last_slice_tokens = tokens
        per_tok = wall_ms / max(tokens, 1)
        self.ewma_tok_ms = self.alpha * per_tok + (1 - self.alpha) * self.ewma_tok_ms
        if wall_ms < self.target_ms * 0.5:
            self.idle_streak = min(self.idle_streak + 1, 8)
        else:
            self.idle_streak = 0
        if wall_ms > self.hard_cap_ms:
            self.ewma_tok_ms = max(self.ewma_tok_ms * 1.25, per_tok)

    @property
    def metrics(self) -> Dict[str, float]:
        if not self.enabled or self.last_slice_tokens <= 0:
            return {}
        return {
            "prefill_slice_tokens": float(self.last_slice_tokens),
            "prefill_slice_ms": float(self.last_slice_ms),
            "prefill_tok_ms_ewma": float(self.ewma_tok_ms),
        }


def _make_prefill_slicer() -> PrefillSlicer:
    enabled = _env_flag("MLXLM_PREFILL_ADAPTIVE")
    target_ms = _env_float("MLXLM_PREFILL_TARGET_MS", 80.0)
    hard_cap_ms = _env_float("MLXLM_PREFILL_HARD_CAP_MS", 250.0)
    min_tokens = _env_int("MLXLM_PREFILL_MIN_TOKENS", 16)
    max_tokens = _env_int("MLXLM_PREFILL_MAX_TOKENS", 256)
    alpha = max(0.05, min(_env_float("MLXLM_PREFILL_EWMA_ALPHA", 0.25), 1.0))
    return PrefillSlicer(
        enabled=enabled,
        target_ms=target_ms,
        hard_cap_ms=hard_cap_ms,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        alpha=alpha,
    )


class SlotLayerSlab:
    """Preallocated storage for a single transformer layer's KV tensors."""

    def __init__(
        self,
        max_slots: int,
        n_kv_heads: int,
        head_dim: int,
        dtype,
        initial_capacity: int = 1,
    ):
        self.max_slots = max_slots
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.capacity = max(initial_capacity, 1)
        shape = (max_slots, self.capacity, n_kv_heads, head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)
        self.lengths: List[int] = [0] * max_slots

    def ensure_capacity(self, required_tokens: int) -> None:
        if required_tokens <= self.capacity:
            return
        grow = max(required_tokens - self.capacity, self.capacity)
        pad_shape = (self.max_slots, grow, self.n_kv_heads, self.head_dim)
        if required_tokens >= 256:
            logging.warning(
                "slot_layer_slab.ensure_capacity layer=%s required=%s current=%s grow=%s dtype=%s",
                id(self),
                required_tokens,
                self.capacity,
                grow,
                self.dtype,
            )
        pad_k = mx.zeros(pad_shape, dtype=self.dtype)
        pad_v = mx.zeros(pad_shape, dtype=self.dtype)
        self.keys = mx.concatenate([self.keys, pad_k], axis=1)
        self.values = mx.concatenate([self.values, pad_v], axis=1)
        self.capacity += grow

    def append_batch(self, slot_ids: Sequence[int], keys, values) -> None:
        if not slot_ids:
            return
        tokens = int(keys.shape[2]) if keys.shape and keys.ndim >= 3 else 0
        if tokens == 0:
            return
        keys_t = mx.transpose(keys, (0, 2, 1, 3))  # [B, T, H, D]
        values_t = mx.transpose(values, (0, 2, 1, 3))
        for idx, slot in enumerate(slot_ids):
            start = self.lengths[slot]
            end = start + tokens
            if end > self.capacity:
                logging.warning(
                    "slot_layer_slab.append_batch slot=%s start=%s tokens=%s end=%s capacity=%s",
                    slot,
                    start,
                    tokens,
                    end,
                    self.capacity,
                )
            self.ensure_capacity(end)
            self.keys[slot, start:end] = keys_t[idx]
            self.values[slot, start:end] = values_t[idx]
            self.lengths[slot] = end

    def gather(self, slot_ids: Sequence[int], max_len: int):
        if not slot_ids:
            empty = mx.zeros((0, self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
            return empty, empty
        if max_len <= 0:
            empty = mx.zeros(
                (len(slot_ids), self.n_kv_heads, 0, self.head_dim), dtype=self.dtype
            )
            return empty, empty
        batch = len(slot_ids)
        keys_full = mx.zeros(
            (batch, max_len, self.n_kv_heads, self.head_dim), dtype=self.dtype
        )
        values_full = mx.zeros(
            (batch, max_len, self.n_kv_heads, self.head_dim), dtype=self.dtype
        )
        for idx, slot in enumerate(slot_ids):
            length = min(self.lengths[slot], max_len)
            if length <= 0:
                continue
            start = length
            keys_slice = self.keys[slot, :length]
            values_slice = self.values[slot, :length]
            keys_full[idx, max_len - length :] = keys_slice
            values_full[idx, max_len - length :] = values_slice
        keys = mx.transpose(keys_full, (0, 2, 1, 3))
        values = mx.transpose(values_full, (0, 2, 1, 3))
        return keys, values

    def reset(self, slot: int) -> None:
        self.lengths[slot] = 0


class SlotKVSlab:
    """Holds per-layer KV slabs covering all available slots."""

    def __init__(self, max_slots: int, initial_capacity: int = 1):
        self.max_slots = max_slots
        self.layers: List[Optional[SlotLayerSlab]] = []
        self.initial_capacity = max(1, int(initial_capacity))

    def ensure_layer(self, layer_idx: int, keys, values) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(None)
        if self.layers[layer_idx] is not None:
            return
        if keys is None or keys.shape[2] == 0:
            return
        n_kv_heads = int(keys.shape[1])
        head_dim = int(keys.shape[3])
        dtype = keys.dtype
        logging.info(
            "slot_layer_slab.ensure_layer layer=%s dtype=%s heads=%s head_dim=%s",
            layer_idx,
            dtype,
            n_kv_heads,
            head_dim,
        )
        initial_capacity = max(int(keys.shape[2]), self.initial_capacity)
        self.layers[layer_idx] = SlotLayerSlab(
            self.max_slots,
            n_kv_heads,
            head_dim,
            dtype,
            initial_capacity=initial_capacity,
        )

    def append_from_cache(
        self, layer_idx: int, slot_id: int, cache, start: int, end: int
    ) -> None:
        state = cache.state
        if isinstance(state, tuple):
            keys, values = state[:2]
        else:
            keys, values = state
        if keys is None or end <= start:
            return
        self.ensure_layer(layer_idx, keys, values)
        layer = self.layers[layer_idx]
        if layer is None:
            return
        slice_keys = keys[..., start:end, :]
        slice_values = values[..., start:end, :]
        layer.append_batch([slot_id], slice_keys, slice_values)

    def append_batch(
        self, layer_idx: int, slot_ids: Sequence[int], keys, values
    ) -> None:
        if not slot_ids:
            return
        self.ensure_layer(layer_idx, keys, values)
        layer = self.layers[layer_idx]
        if layer is None:
            return
        # TODO: Replace append with paged KV updates when paged attention is wired in.
        layer.append_batch(slot_ids, keys, values)

    def reset_slot(self, slot: int) -> None:
        for layer in self.layers:
            if layer is not None:
                layer.reset(slot)


class SlotBatchCache:
    """Batch-aware cache view backed by SlotKVSlab."""

    def __init__(self, slab: SlotKVSlab, layer_idx: int):
        self.slab = slab
        self.layer_idx = layer_idx
        self.slot_ids: tuple[int, ...] = ()
        self.left_padding = mx.zeros((0,), dtype=mx.int32)
        self.offset = mx.zeros((0,), dtype=mx.int32)
        self._contexts: list = []
        self._lengths: List[int] = []
        self._max_len = 0
        self._capacity = 0
        self._scratch_keys = None
        self._scratch_values = None
        self._scratch_lengths: List[int] = []
        self._last_slots: tuple[int, ...] = ()

    def _derive_sample_shape(self, keys):
        if keys is None:
            return mx.float32, 1, 1
        dtype = keys.dtype
        if hasattr(keys, "shape") and len(keys.shape) >= 4:
            return dtype, int(keys.shape[1]), int(keys.shape[-1])
        return dtype, 1, 1

    def _apply_lengths(self):
        self._max_len = max(self._scratch_lengths, default=0)
        if self._scratch_lengths:
            left_padding = [self._max_len - length for length in self._scratch_lengths]
            self.left_padding = mx.array(left_padding, dtype=mx.int32)
            offsets = [
                lp + length for lp, length in zip(left_padding, self._scratch_lengths)
            ]
            self.offset = mx.array(offsets, dtype=mx.int32)
        else:
            self.left_padding = mx.zeros((0,), dtype=mx.int32)
            self.offset = mx.zeros((0,), dtype=mx.int32)
        self._lengths = list(self._scratch_lengths)

    def _rebuild_scratch(self, layer, lengths, sample_keys=None):
        batch = len(self.slot_ids)
        if layer is not None:
            capacity = layer.capacity
            dtype = layer.dtype
            heads = layer.n_kv_heads
            head_dim = layer.head_dim
        else:
            capacity = max(max(lengths, default=0), 1)
            dtype, heads, head_dim = self._derive_sample_shape(sample_keys)
        shape = (batch, capacity, heads, head_dim)
        self._scratch_keys = mx.zeros(shape, dtype=dtype)
        self._scratch_values = mx.zeros(shape, dtype=dtype)
        self._capacity = capacity
        self._scratch_lengths = [0] * batch
        if layer is not None:
            for idx, (slot, length) in enumerate(zip(self.slot_ids, lengths)):
                if length <= 0:
                    continue
                data_k = layer.keys[slot, :length]
                data_v = layer.values[slot, :length]
                start = capacity - length
                self._scratch_keys[idx, start:capacity] = data_k
                self._scratch_values[idx, start:capacity] = data_v
                self._scratch_lengths[idx] = length
        else:
            for idx, length in enumerate(lengths):
                self._scratch_lengths[idx] = length
        self._last_slots = self.slot_ids
        self._apply_lengths()

    def bind(self, contexts: Sequence) -> None:
        self.slot_ids = tuple(ctx.state.slot_id for ctx in contexts)
        self._contexts = list(contexts)

        if not contexts:
            self._scratch_keys = None
            self._scratch_values = None
            self._scratch_lengths = []
            self._capacity = 0
            self._last_slots = ()
            self._lengths = []
            self._max_len = 0
            self.left_padding = mx.zeros((0,), dtype=mx.int32)
            self.offset = mx.zeros((0,), dtype=mx.int32)
            return

        layer_cache = contexts[0].prompt_cache[self.layer_idx]
        state = layer_cache.state
        if isinstance(state, tuple):
            sample_keys = state[0]
        else:
            sample_keys = state
        keys, values = state[:2] if isinstance(state, tuple) else (state, None)
        self.slab.ensure_layer(self.layer_idx, keys, values)
        layer = self.slab.layers[self.layer_idx]

        new_lengths = [ctx._kv_lengths[self.layer_idx] for ctx in contexts]
        capacity_target = (
            layer.capacity if layer is not None else max(max(new_lengths, default=0), 1)
        )
        needs_rebuild = (
            self._scratch_keys is None
            or self._scratch_values is None
            or self._capacity != capacity_target
            or len(self._scratch_lengths) != len(new_lengths)
            or self._last_slots != self.slot_ids
        )

        if needs_rebuild:
            self._rebuild_scratch(layer, new_lengths, sample_keys)
        else:
            resize_needed = False
            for length in new_lengths:
                if length > self._capacity:
                    resize_needed = True
                    break
            if resize_needed:
                self._rebuild_scratch(layer, new_lengths, sample_keys)
            else:
                for idx, length in enumerate(new_lengths):
                    self._scratch_lengths[idx] = length
                self._apply_lengths()

    def _gather(self):
        if (
            self._scratch_keys is None
            or self._scratch_values is None
            or not self.slot_ids
        ):
            empty = mx.zeros((0, 0, 0, 0), dtype=mx.float32)
            offsets = mx.zeros((0,), dtype=mx.int32)
            return empty, empty, offsets, self.left_padding
        start = self._capacity - self._max_len
        gathered_k = self._scratch_keys[:, start : self._capacity]
        gathered_v = self._scratch_values[:, start : self._capacity]
        keys = mx.transpose(gathered_k, (0, 2, 1, 3))
        values = mx.transpose(gathered_v, (0, 2, 1, 3))
        return keys, values, self.offset, self.left_padding

    def update_and_fetch(self, keys, values):
        tokens = int(keys.shape[2]) if keys.shape and keys.ndim >= 3 else 0
        if tokens:
            self.slab.append_batch(self.layer_idx, self.slot_ids, keys, values)
            lengths_before = [ctx._kv_lengths[self.layer_idx] for ctx in self._contexts]
            lengths_after = [length + tokens for length in lengths_before]
            layer = self.slab.layers[self.layer_idx]
            rebuild_needed = (
                self._scratch_keys is None
                or self._scratch_values is None
                or len(self._scratch_lengths) != len(lengths_after)
                or self._last_slots != self.slot_ids
                or (layer is not None and layer.capacity != self._capacity)
            )
            if rebuild_needed:
                self._rebuild_scratch(layer, lengths_after)
            else:
                keys_t = mx.transpose(keys, (0, 2, 1, 3))
                values_t = mx.transpose(values, (0, 2, 1, 3))
                for idx in range(len(self.slot_ids)):
                    prev_len = (
                        self._scratch_lengths[idx]
                        if idx < len(self._scratch_lengths)
                        else 0
                    )
                    new_len = lengths_after[idx]
                    delta = new_len - prev_len
                    if delta <= 0:
                        continue
                    start = self._capacity - new_len
                    end = self._capacity - prev_len
                    if end > start:
                        self._scratch_keys[idx, start:end] = keys_t[idx]
                        self._scratch_values[idx, start:end] = values_t[idx]
                    self._scratch_lengths[idx] = new_len
                self._apply_lengths()
            for idx, ctx in enumerate(self._contexts):
                ctx._kv_lengths[self.layer_idx] = lengths_after[idx]
                ctx.prompt_cache[self.layer_idx].update_and_fetch(
                    keys[idx : idx + 1], values[idx : idx + 1]
                )
        keys_out, values_out, _, _ = self._gather()
        return keys_out, values_out

    @property
    def state(self):
        return self._gather()


class PagedBatchCache:
    """Minimal batch cache that proxies updates directly into paged adapters."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self._contexts: List[SequenceContext] = []

    def bind(self, contexts: Sequence[SequenceContext]) -> None:
        self._contexts = [ctx for ctx in contexts if not ctx.state.finished]

    def update_and_fetch(self, keys, values):
        if not self._contexts:
            return keys, values
        token_count = int(keys.shape[2]) if keys is not None and keys.ndim >= 3 else 0
        for idx, ctx in enumerate(self._contexts):
            cache = ctx.prompt_cache[self.layer_idx]
            if token_count == 1 and hasattr(cache, "append_token"):
                k_slice = keys[idx, :, 0]
                v_slice = values[idx, :, 0]
                cache.append_token(k_slice, v_slice)
            else:
                cache.update_and_fetch(keys[idx : idx + 1], values[idx : idx + 1])
            if hasattr(ctx, "_kv_lengths"):
                ctx._kv_lengths[self.layer_idx] = getattr(
                    cache, "offset", ctx._kv_lengths[self.layer_idx]
                )
        return keys, values

    def make_mask(self, *args, **kwargs):
        if not self._contexts:
            return None
        cache = self._contexts[0].prompt_cache[self.layer_idx]
        maker = getattr(cache, "make_mask", None)
        if maker is None:
            return None
        return maker(*args, **kwargs)

    @property
    def state(self):
        return ()

    @property
    def offset(self) -> int:
        if not self._contexts:
            return 0
        cache = self._contexts[0].prompt_cache[self.layer_idx]
        return int(getattr(cache, "offset", 0))


class SlotGenerator:
    """Slot-indexed generator that batches model forwards for decode."""

    def __init__(
        self,
        model,
        tokenizer,
        slot_alloc,
        *,
        prefill_chunk: int = 1024,
        decode_engine: str = "dense",
        prefill_ramp_chunk: Optional[int] = None,
        prefill_hybrid_threshold: int = 0,
        prefill_ramp_budget_ms: Optional[float] = None,
        arrays_provider_cls: Optional[type] = None,
        array_decode_runner_cls: Optional[type] = None,
        prefill_array_runner_cls: Optional[type] = None,
    ):
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.slot_alloc = slot_alloc
        self.prefill_chunk = prefill_chunk
        self.prefill_ramp_chunk = self._normalize_ramp_chunk(prefill_ramp_chunk)
        self.prefill_hybrid_dense_tokens = max(0, int(prefill_hybrid_threshold or 0))
        if self.prefill_hybrid_dense_tokens > 0 and self.prefill_ramp_chunk > 0:
            self.prefill_hybrid_dense_tokens = min(
                self.prefill_hybrid_dense_tokens, self.prefill_ramp_chunk
            )
        budget = (
            float(prefill_ramp_budget_ms)
            if prefill_ramp_budget_ms and prefill_ramp_budget_ms > 0
            else None
        )
        self._prefill_ramp_budget_ms: Optional[float] = budget
        self._prefill_ms_per_token: Optional[float] = None
        self._adaptive_ramp_tokens: Optional[int] = self.prefill_ramp_chunk
        self._ramp_min_tokens = (
            0 if self.prefill_ramp_chunk <= 0 else min(self.prefill_ramp_chunk, 8)
        )
        self._slab: Optional[SlotKVSlab] = None
        self._batched_cache_layers: list[SlotBatchCache] = []
        self._paged_batch_layers: list[PagedBatchCache] = []
        self._last_decode_profile = {}
        self._compiled_decode = None
        self.paged_cache = None
        self.kv_head_mapping = None
        self._paged_manager = None
        self._paged_num_layers = None
        self.prefix_cache = None
        self._paged_model_signature = None
        self.kv_quant_mode = None
        self.prefill_slicer = _make_prefill_slicer()
        self._prefill_metrics: Dict[str, float] = {}
        self._compile_cache: Dict[
            Tuple[int, int, int, int, int, int, str, Optional[str]], object
        ] = {}
        self._compile_cache_hits = 0
        self._compile_cache_misses = 0
        self._arrays_provider_cls = arrays_provider_cls or PagedArraysProvider
        self._array_decode_runner_cls = array_decode_runner_cls or ArrayDecodeRunner
        self._array_decode_runner_engines = getattr(
            self._array_decode_runner_cls, "SUPPORTED_ENGINES", {"dense"}
        )
        self._prefill_array_runner_cls = prefill_array_runner_cls or PrefillArrayRunner
        if decode_engine is None:
            decode_engine = "dense"
        engines = self._array_decode_runner_engines
        decode_choice = decode_engine if decode_engine in engines else "dense"
        if decode_choice != decode_engine:
            logging.warning(
                "SlotGenerator: decode_engine '%s' is unavailable; falling back to '%s'",
                decode_engine,
                decode_choice,
            )
        self.decode_engine = decode_choice
        self.use_array_decode = self.decode_engine in engines
        array_prefill_default = (
            self.decode_engine in engines
            and self.decode_engine.startswith("paged-arrays")
        )
        env_flag = os.environ.get("MLXLM_PREFILL_ARRAY")
        if env_flag is not None:
            self._array_prefill_flag = _env_flag("MLXLM_PREFILL_ARRAY")
        else:
            self._array_prefill_flag = array_prefill_default
        self._fast_available = hasattr(mx, "fast")
        self.use_array_prefill = bool(self._array_prefill_flag and self._fast_available)
        self._arrays_provider: Optional[PagedArraysProvider] = None
        self._paged_backend = None
        self.array_decode_runner: Optional[ArrayDecodeRunner] = None
        self.prefill_array_runner: Optional[PrefillArrayRunner] = None
        self.kv_block_size = prefill_chunk
        self._array_prefill_first_chunk_ms = 0.0

    # ------------------------------------------------------------------#
    # Admission / release
    # ------------------------------------------------------------------#
    def on_admit(self, ctx) -> None:
        if getattr(ctx.state, "slot_id", None) is None:
            slot = self.slot_alloc.alloc()
            if slot is None:
                return False
            ctx.state.slot_id = slot
        prompt_tokens = getattr(ctx, "prompt_tokens", [])
        if not hasattr(ctx, "prompt_ids_np"):
            ctx.prompt_ids_np = np.array(prompt_tokens, dtype=np.int32)
        if getattr(ctx, "prompt_cache", None) is None:
            ctx.prompt_cache = make_prompt_cache(self.model)
        if not hasattr(ctx, "_cache_id"):
            ctx._cache_id = id(ctx.prompt_cache)
        if not hasattr(ctx, "_kv_lengths"):
            ctx._kv_lengths = [0] * len(ctx.prompt_cache)
        ctx.slot_initialized = True
        if prompt_tokens:
            ctx.last_token_id = int(prompt_tokens[-1])
        else:
            bos = getattr(self.tokenizer, "bos_token_id", 0)
            ctx.last_token_id = int(bos)
        if (
            self.paged_cache is not None
            and self._paged_manager is not None
            and self._paged_num_layers is not None
        ):
            seq_id = self.paged_cache.register(
                ctx.state.request_id, ctx.state.prompt_len
            )
            ctx.state.paged_seq_id = seq_id
            paged_caches: List[PagedKVCacheAdapter] = []
            for layer_idx in range(len(ctx.prompt_cache)):
                paged_caches.append(
                    PagedKVCacheAdapter(self._paged_manager, seq_id, layer_idx)
                )
            ctx.prompt_cache = paged_caches
        ramp_tokens = self._current_ramp_limit()
        ctx.state.prefill_ramp_remaining = ramp_tokens
        hybrid_remaining = max(0, int(getattr(ctx.state, "hybrid_dense_remaining", 0)))
        if hybrid_remaining > 0 and ramp_tokens > 0:
            ctx.state.hybrid_dense_remaining = min(hybrid_remaining, ramp_tokens)
        ctx.state.prefill_decode_ready = False
        return True

    def on_release(self, ctx) -> None:
        slot_id = getattr(ctx.state, "slot_id", None)
        if slot_id is not None:
            self.slot_alloc.release(slot_id)
            if self._slab is not None:
                self._slab.reset_slot(slot_id)
        ctx.state.slot_id = None
        ctx.slot_initialized = False
        seq_id = getattr(ctx.state, "paged_seq_id", None)
        if self.paged_cache is not None and seq_id is not None:
            self.paged_cache.release(ctx.state.request_id)
        if self.prefill_array_runner is not None and seq_id is not None:
            self.prefill_array_runner.release_overlays([seq_id])
        ctx.state.paged_seq_id = None
        ctx.state.prefill_decode_ready = False
        ctx.state.prefill_ramp_remaining = 0

    def _prefill_warm_batches(self) -> list[int]:
        capacity = getattr(self.slot_alloc, "capacity", 1) or 1
        cap = max(1, min(int(capacity), 8))
        batches = []
        value = 1
        while value < cap:
            batches.append(value)
            value <<= 1
        if not batches or batches[-1] != cap:
            batches.append(cap)
        return batches

    def _prefill_warm_chunks(self) -> list[int]:
        target = max(1, int(self.prefill_chunk or 1))
        chunks = []
        value = 1
        while value < target:
            chunks.append(value)
            value <<= 1
        if not chunks or chunks[-1] != target:
            chunks.append(target)
        if self.prefill_ramp_chunk and self.prefill_ramp_chunk not in chunks:
            chunks.append(self.prefill_ramp_chunk)
        return sorted({int(max(1, chunk)) for chunk in chunks})

    def _update_ramp_profile(self, tokens: int, model_ms: float) -> None:
        if (
            self._prefill_ramp_budget_ms is None
            or tokens <= 0
            or model_ms <= 0.0
            or self.prefill_ramp_chunk <= 0
        ):
            return
        ms_per_token = model_ms / max(1, tokens)
        if self._prefill_ms_per_token is None:
            self._prefill_ms_per_token = ms_per_token
        else:
            alpha = 0.3
            self._prefill_ms_per_token = (
                alpha * ms_per_token + (1.0 - alpha) * self._prefill_ms_per_token
            )
        estimate = max(self._prefill_ms_per_token, 1e-3)
        target_tokens = int(self._prefill_ramp_budget_ms / estimate)
        base = max(1, int(self.prefill_ramp_chunk or 1))
        min_tokens = self._ramp_min_tokens or 1
        adaptive = max(min_tokens, min(base, target_tokens))
        self._adaptive_ramp_tokens = adaptive

    def _normalize_ramp_chunk(self, ramp_value: Optional[int]) -> int:
        if ramp_value is None:
            return min(self.prefill_chunk, 64)
        ramp = int(ramp_value)
        if ramp <= 0:
            return 0
        return max(1, min(ramp, self.prefill_chunk))

    def _current_ramp_limit(self) -> int:
        base = max(0, int(self.prefill_ramp_chunk or 0))
        if base <= 0:
            return 0
        adaptive = self._adaptive_ramp_tokens
        if adaptive is None:
            adaptive = base
        min_tokens = self._ramp_min_tokens or 1
        return max(min_tokens, min(base, adaptive))

    def _bucket_prefill_len(self, requested: int) -> int:
        limit = max(1, int(self.prefill_chunk or 1))
        value = max(1, min(int(requested), limit))
        bucket = 1
        while (bucket << 1) <= value:
            bucket <<= 1
        return bucket

    def _apply_ramp_chunk(self, ctx, chunk_len: int) -> int:
        ramp = getattr(self, "prefill_ramp_chunk", 0)
        if not ramp:
            return chunk_len
        consumed = max(0, int(getattr(ctx.state, "prompt_pos", 0)))
        if consumed >= ramp:
            return chunk_len
        remaining_ramp = max(ramp - consumed, 1)
        return max(1, min(chunk_len, remaining_ramp))

    def _limit_paged_chunk_len(self, ctx, chunk_len: int) -> int:
        ramp_remaining = max(0, int(getattr(ctx.state, "prefill_ramp_remaining", 0)))
        if ramp_remaining <= 0:
            return chunk_len
        return max(1, min(chunk_len, ramp_remaining))

    def _maybe_mark_ramp_ready(self, ctx) -> None:
        if getattr(ctx.state, "prefill_decode_ready", False):
            return
        if getattr(self, "prefill_ramp_chunk", 0) <= 0:
            return
        ramp_remaining = int(getattr(ctx.state, "prefill_ramp_remaining", 0))
        if ramp_remaining > 0:
            return
        ctx.state.prefill_decode_ready = True

    def set_paged_backend(
        self,
        paged_cache,
        kv_head_mapping,
        *,
        manager,
        num_layers,
        prefix_cache=None,
        model_signature=None,
        kv_quant_mode=None,
    ) -> None:
        self.paged_cache = paged_cache
        self.kv_head_mapping = kv_head_mapping
        self._paged_manager = manager
        self._paged_num_layers = num_layers
        self.prefix_cache = prefix_cache
        self._paged_model_signature = model_signature
        self.kv_quant_mode = kv_quant_mode
        array_engine = (
            self.decode_engine in self._array_decode_runner_engines
            and hasattr(mx.fast, "_paged_attention_with_overlay_impl")
        )
        self.use_array_decode = bool(array_engine and paged_cache and manager)
        self.array_decode_runner = None
        self.prefill_array_runner = None
        self._arrays_provider = None
        self._paged_backend = None
        logging.info(
            "SlotGenerator: set_paged_backend use_array_decode=%s "
            "array_engine=%s overlay_op=%s paged_cache=%s manager=%s",
            self.use_array_decode,
            array_engine,
            hasattr(mx.fast, "_paged_attention_with_overlay_impl"),
            bool(paged_cache),
            bool(manager),
        )
        if self.use_array_decode:
            try:
                provider_cls = self._arrays_provider_cls
                self._arrays_provider = provider_cls(
                    manager, num_layers=num_layers, block_size=self.kv_block_size
                )
            except Exception:
                logging.warning(
                    "SlotGenerator: unable to initialize array provider",
                    exc_info=True,
                )
                self._arrays_provider = None
                self.array_decode_runner = None
                self.use_array_prefill = False
                self.prefill_array_runner = None
            else:
                try:
                    runner_cls = self._array_decode_runner_cls
                    self.array_decode_runner = runner_cls(
                        self.model, manager, decode_engine=self.decode_engine
                    )
                    self.use_array_prefill = bool(
                        self._array_prefill_flag and self._fast_available
                    )
                    if self.use_array_prefill:
                        prefill_cls = self._prefill_array_runner_cls
                        self.prefill_array_runner = prefill_cls(
                            self.array_decode_runner
                        )
                        try:
                            warm_batch = tuple(self._prefill_warm_batches())
                            warm_chunks = tuple(self._prefill_warm_chunks())
                            self.prefill_array_runner.warmup(
                                batch_size=warm_batch,
                                chunk_len=warm_chunks,
                            )
                        except Exception:
                            logging.debug(
                                "SlotGenerator: prefill runner warmup failed",
                                exc_info=True,
                            )
                    logging.info(
                        "SlotGenerator: array decode runner initialized (prefill=%s)",
                        self.use_array_prefill,
                    )
                except Exception:
                    logging.warning(
                        "SlotGenerator: unable to initialize array decode runner",
                        exc_info=True,
                    )
                    self.array_decode_runner = None
                    self.prefill_array_runner = None
                    self.use_array_prefill = False
        if not self.use_array_decode:
            self.use_array_prefill = False
        if not self.use_array_decode and self.paged_cache is not None:
            from .attn_backend import PagedAttentionBackend

            try:
                self._paged_backend = PagedAttentionBackend(
                    self.paged_cache, kv_head_mapping=self.kv_head_mapping
                )
            except Exception:
                logging.debug(
                    "SlotGenerator: unable to initialize paged attention backend",
                    exc_info=True,
                )
                self._paged_backend = None

    # ------------------------------------------------------------------#
    # Prefill helpers
    # ------------------------------------------------------------------#
    def prefill_step(self, contexts: Sequence) -> None:
        self._prefill_metrics = {}
        computed = False
        for ctx in contexts:
            processed = self.prefill_tokens(ctx, self.prefill_chunk)
            computed = computed or processed > 0
        if computed:
            mx.eval()

    def prefill_tokens(self, ctx, max_tokens: int) -> int:
        if getattr(ctx.state, "finished", False):
            return 0
        self._ensure_prompt_cache(ctx)
        self._maybe_reuse_prefix(ctx)
        remaining = ctx.state.remaining_prompt_tokens
        if remaining <= 0 or max_tokens <= 0:
            self._record_apc_prefixes(ctx)
            return 0
        take = self._bucket_prefill_len(min(max_tokens, remaining))
        take = self._apply_ramp_chunk(ctx, take)
        hybrid_remaining = 0
        if self.prefill_hybrid_dense_tokens > 0:
            hybrid_remaining = max(
                0, int(getattr(ctx.state, "hybrid_dense_remaining", 0))
            )
            if hybrid_remaining > 0:
                take = min(take, hybrid_remaining)
        start = ctx.state.prompt_pos
        span = ctx.prompt_ids_np[start : start + take]
        if span.size == 0:
            ctx.state.prompt_pos = start + take
            return 0
        paged_prefill = None
        view = None
        if (
            self.paged_cache is not None
            and getattr(ctx.state, "paged_seq_id", None) is not None
            and take > 0
        ):
            paged_prefill = self.paged_cache.begin_prefill(
                ctx.state.request_id,
                take,
                kv_head_mapping=self.kv_head_mapping,
            )
            view = paged_prefill.view
            if _PAGED_TRACE and view is not None:
                _PAGED_LOG.info(
                    "slot.prefill.begin request=%s seq_id=%s take=%s base_lens=%s context=%s tables_shape=%s",
                    getattr(ctx.state, "request_id", "unknown"),
                    paged_prefill.seq_id,
                    take,
                    _array_preview(getattr(view, "prefill_base_lens", None)),
                    _array_preview(view.context_lens),
                    tuple(view.block_tables.shape),
                )
        slice_count = 0
        processed = 0
        total_start = time.perf_counter()
        while processed < take:
            chunk_len = take - processed
            if view is None and self.prefill_slicer.enabled:
                chunk_len = self.prefill_slicer.next_length(chunk_len)
            chunk = span[processed : processed + chunk_len]
            if chunk.size == 0:
                break
            inputs = np.expand_dims(chunk, axis=0)
            slice_start = time.perf_counter()
            if view is not None:
                with batch_view_scope(view):
                    self.model(_mx_array(inputs), cache=ctx.prompt_cache)
            else:
                self.model(_mx_array(inputs), cache=ctx.prompt_cache)
            slice_ms = (time.perf_counter() - slice_start) * 1000.0
            processed += chunk_len
            slice_count += 1
            ctx.state.prompt_pos = start + processed
            ctx.last_token_id = int(chunk[-1])
            if view is None and self.prefill_slicer.enabled:
                self.prefill_slicer.observe(chunk_len, slice_ms)
                self._prefill_metrics.update(self.prefill_slicer.metrics)
            if _BENCH_TRACE:
                backend = getattr(self.model, "__class__", type(self.model)).__name__
                _BENCH_LOG.info(
                    "slot.prefill.slice request=%s tokens=%s remaining=%s paged=%s backend=%s slice_ms=%.3f",
                    getattr(ctx.state, "request_id", "unknown"),
                    chunk_len,
                    take - processed,
                    view is not None,
                    backend,
                    slice_ms,
                )

        if paged_prefill is not None:
            paged_prefill.commit()
            if _PAGED_TRACE:
                _PAGED_LOG.info(
                    "slot.prefill.commit request=%s seq_id=%s tokens=%s slices=%s",
                    getattr(ctx.state, "request_id", "unknown"),
                    paged_prefill.seq_id,
                    processed,
                    slice_count,
                )
        self._sync_prompt_cache(ctx)
        self._record_apc_prefixes(ctx)
        ramp_remaining = int(getattr(ctx.state, "prefill_ramp_remaining", 0))
        if ramp_remaining > 0:
            ctx.state.prefill_ramp_remaining = max(0, ramp_remaining - processed)
            ramp_remaining = ctx.state.prefill_ramp_remaining
        if ramp_remaining <= 0:
            self._maybe_mark_ramp_ready(ctx)
        if self.paged_cache is not None:
            self._refresh_paged_lengths(ctx)
        if hybrid_remaining > 0:
            ctx.state.hybrid_dense_remaining = max(0, hybrid_remaining - processed)
            self._prefill_metrics["prefill_hybrid_dense_tokens"] = (
                self._prefill_metrics.get("prefill_hybrid_dense_tokens", 0.0)
                + processed
            )
            duration_ms = (time.perf_counter() - total_start) * 1000.0
            self._prefill_metrics["prefill_hybrid_dense_ms"] = (
                self._prefill_metrics.get("prefill_hybrid_dense_ms", 0.0) + duration_ms
            )
        if _BENCH_TRACE:
            backend = getattr(self.model, "__class__", type(self.model)).__name__
            _BENCH_LOG.info(
                "slot.prefill request=%s take=%s remaining=%s paged=%s backend=%s duration_ms=%.3f slices=%s",
                getattr(ctx.state, "request_id", "unknown"),
                processed,
                ctx.state.remaining_prompt_tokens,
                view is not None,
                backend,
                (time.perf_counter() - total_start) * 1000.0,
                slice_count,
            )
        return processed

    def prefill_tokens_multi(
        self, contexts: Sequence[SequenceContext], chunk_len: int
    ) -> List[int]:
        chunk_len = self._bucket_prefill_len(chunk_len)
        if chunk_len <= 0 or not contexts:
            return [0 for _ in contexts]
        processed = [0 for _ in contexts]
        paged_entries: List[tuple[int, SequenceContext, int]] = []
        fallback_entries: List[tuple[int, SequenceContext, int]] = []
        for idx, ctx in enumerate(contexts):
            if getattr(ctx.state, "finished", False):
                continue
            self._ensure_prompt_cache(ctx)
            self._maybe_reuse_prefix(ctx)
            remaining = ctx.state.remaining_prompt_tokens
            if remaining <= 0:
                self._record_apc_prefixes(ctx)
                continue
            take = self._bucket_prefill_len(min(chunk_len, remaining))
            take = self._apply_ramp_chunk(ctx, take)
            if take <= 0:
                continue
            hybrid_remaining = 0
            if self.prefill_hybrid_dense_tokens > 0:
                hybrid_remaining = max(
                    0, int(getattr(ctx.state, "hybrid_dense_remaining", 0))
                )
            if hybrid_remaining > 0:
                dense_take = min(take, hybrid_remaining)
                fallback_entries.append((idx, ctx, dense_take))
                continue
            if (
                self.paged_cache is not None
                and getattr(ctx.state, "paged_seq_id", None) is not None
            ):
                paged_entries.append((idx, ctx, take))
            else:
                fallback_entries.append((idx, ctx, take))

        for idx, ctx, take in fallback_entries:
            processed[idx] += self.prefill_tokens(ctx, take)

        if paged_entries:
            paged_counts = self._prefill_tokens_array_batch(paged_entries, chunk_len)
            if paged_counts is None:
                paged_counts = self._prefill_tokens_paged_batch(
                    paged_entries, chunk_len
                )
            for (idx, _ctx, _take), count in zip(paged_entries, paged_counts):
                processed[idx] += count

        return processed

    def _prefill_tokens_paged_batch(
        self, entries: Sequence[tuple[int, SequenceContext, int]], chunk_len: int
    ) -> List[int]:
        if not entries:
            return []
        if self.paged_cache is None:
            raise RuntimeError("paged prefill batch requires paged cache")
        takes = []
        for _, ctx, take in entries:
            limit = min(take, chunk_len)
            limit = self._limit_paged_chunk_len(ctx, limit)
            takes.append(self._bucket_prefill_len(limit))
        effective_len = min(takes)
        if effective_len <= 0:
            return [0 for _ in entries]
        contexts = [ctx for _, ctx, _ in entries]
        request_ids = [ctx.state.request_id for ctx in contexts]
        handle = self.paged_cache.begin_prefill_many(
            request_ids,
            [effective_len] * len(contexts),
            kv_head_mapping=self.kv_head_mapping,
        )
        view = handle.view
        layer_count = len(contexts[0].prompt_cache)
        batch_caches = self._ensure_paged_batch_layers(layer_count)
        for cache in batch_caches:
            cache.bind(contexts)
        inputs = []
        for ctx in contexts:
            start = ctx.state.prompt_pos
            chunk = ctx.prompt_ids_np[start : start + effective_len]
            if chunk.shape[0] != effective_len:
                raise ValueError(
                    f"prefill chunk mismatch for request {ctx.state.request_id}"
                )
            inputs.append(chunk)
        tokens_np = np.stack(inputs, axis=0)
        slice_start = time.perf_counter()
        inputs = mx.array(tokens_np, dtype=mx.int32)
        self._run_with_backend(
            view,
            lambda: self._call_model(inputs, batch_caches, view),
        )
        model_ms = (time.perf_counter() - slice_start) * 1000.0
        slice_ms = model_ms
        commit_start = time.perf_counter()
        handle.commit()
        commit_ms = (time.perf_counter() - commit_start) * 1000.0
        total_slice_s = max(slice_ms, 0.0) / 1000.0
        total_model_s = max(model_ms, 0.0) / 1000.0
        total_commit_s = max(commit_ms, 0.0) / 1000.0
        self._update_ramp_profile(effective_len, model_ms)
        self._prefill_metrics["paged_prefill_slice_s"] = (
            self._prefill_metrics.get("paged_prefill_slice_s", 0.0) + total_slice_s
        )
        self._prefill_metrics["paged_prefill_model_s"] = (
            self._prefill_metrics.get("paged_prefill_model_s", 0.0) + total_model_s
        )
        self._prefill_metrics["paged_prefill_commit_s"] = (
            self._prefill_metrics.get("paged_prefill_commit_s", 0.0) + total_commit_s
        )
        self._prefill_metrics["paged_prefill_slice_count"] = (
            self._prefill_metrics.get("paged_prefill_slice_count", 0.0) + 1.0
        )
        if self._prefill_metrics.get("paged_prefill_first_slice_ms", 0.0) <= 0.0:
            self._prefill_metrics["paged_prefill_first_slice_ms"] = max(slice_ms, 0.0)
        for row, ctx in enumerate(contexts):
            ctx.state.prompt_pos += effective_len
            ctx.last_token_id = int(tokens_np[row, effective_len - 1])
            self._sync_prompt_cache(ctx)
            self._record_apc_prefixes(ctx)
            ramp_remaining = int(getattr(ctx.state, "prefill_ramp_remaining", 0))
            if ramp_remaining > 0:
                ctx.state.prefill_ramp_remaining = max(
                    0, ramp_remaining - effective_len
                )
                ramp_remaining = ctx.state.prefill_ramp_remaining
            if ramp_remaining <= 0:
                self._maybe_mark_ramp_ready(ctx)
            if self.paged_cache is not None:
                self._refresh_paged_lengths(ctx)
            # Skip slicer feedback for paged batch runs; chunks are already fixed.
            if _BENCH_TRACE:
                backend = getattr(self.model, "__class__", type(self.model)).__name__
                _BENCH_LOG.info(
                    "slot.prefill.slice request=%s tokens=%s remaining=%s paged=%s backend=%s slice_ms=%.3f",
                    getattr(ctx.state, "request_id", "unknown"),
                    effective_len,
                    ctx.state.remaining_prompt_tokens,
                    True,
                    backend,
                    slice_ms,
                )
        return [effective_len for _ in entries]

    def _prefill_tokens_array_batch(
        self, entries: Sequence[tuple[int, SequenceContext, int]], chunk_len: int
    ) -> Optional[List[int]]:
        if (
            not entries
            or self.prefill_array_runner is None
            or not self.use_array_prefill
        ):
            return None
        if self._arrays_provider is None:
            return None
        # Group entries by their effective chunk length so we can batch
        # heterogeneous tails without falling back.
        grouped: Dict[int, list[tuple[int, SequenceContext, int]]] = {}
        for entry in entries:
            idx, ctx, take = entry
            seq_id = getattr(ctx.state, "paged_seq_id", None)
            if seq_id is None:
                return None
            limit = min(chunk_len, take)
            limit = self._limit_paged_chunk_len(ctx, limit)
            effective_len = self._bucket_prefill_len(limit)
            if effective_len <= 0:
                continue
            grouped.setdefault(effective_len, []).append(entry)
        if not grouped:
            return [0 for _ in entries]
        processed = {idx: 0 for idx, _, _ in entries}
        for effective_len, group_entries in grouped.items():
            result = self._run_array_prefill_group(group_entries, effective_len)
            if result is None:
                return None
            for idx, count in result.items():
                processed[idx] = count
        return [processed.get(idx, 0) for idx, _, _ in entries]

    def _run_array_prefill_group(
        self,
        group_entries: Sequence[tuple[int, SequenceContext, int]],
        chunk_len: int,
    ) -> Optional[Dict[int, int]]:
        seq_ids = []
        contexts = []
        tokens_list = []
        entry_indices: List[int] = []
        for idx, ctx, _take in group_entries:
            seq_id = getattr(ctx.state, "paged_seq_id", None)
            if seq_id is None:
                return None
            start = ctx.state.prompt_pos
            chunk = ctx.prompt_ids_np[start : start + chunk_len]
            if chunk.shape[0] != chunk_len:
                return None
            seq_ids.append(seq_id)
            contexts.append(ctx)
            tokens_list.append(chunk)
            entry_indices.append(idx)
        try:
            view = self.paged_cache.make_batch_view(
                seq_ids, kv_head_mapping=self.kv_head_mapping
            )
        except Exception:
            return None
        array_view = self._maybe_array_view(view, max_steps=chunk_len)
        if not isinstance(array_view, ArrayBatchView):
            return None
        tokens_np = np.stack(tokens_list, axis=0)
        try:
            written = self.prefill_array_runner.prefill_chunk(
                mx.array(tokens_np, dtype=mx.int32),
                array_view,
                chunk_len=chunk_len,
            )
            self._accumulate_array_prefill_metrics(
                self.prefill_array_runner.consume_metrics()
            )
        except Exception:
            logging.debug(
                "SlotGenerator: array prefill chunk failed; falling back",
                exc_info=True,
            )
            self.prefill_array_runner.consume_metrics()
            return None
        if written <= 0:
            self.prefill_array_runner.consume_metrics()
            return None
        for row, ctx in enumerate(contexts):
            ctx.state.prompt_pos += chunk_len
            ctx.last_token_id = int(tokens_np[row, chunk_len - 1])
            self._sync_prompt_cache(ctx)
            self._record_apc_prefixes(ctx)
            ramp_remaining = int(getattr(ctx.state, "prefill_ramp_remaining", 0))
            if ramp_remaining > 0:
                ctx.state.prefill_ramp_remaining = max(0, ramp_remaining - chunk_len)
                ramp_remaining = ctx.state.prefill_ramp_remaining
            if ramp_remaining <= 0:
                self._maybe_mark_ramp_ready(ctx)
            if self.paged_cache is not None:
                self._refresh_paged_lengths(ctx)
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "slot.prefill.array request_ids=%s tokens=%s",
                _seq_preview([ctx.state.request_id for ctx in contexts]),
                chunk_len,
            )
        return {idx: chunk_len for idx in entry_indices}

    def _maybe_reuse_prefix(self, ctx) -> int:
        if (
            self.prefix_cache is None
            or getattr(ctx.state, "paged_seq_id", None) is None
            or not getattr(ctx.state, "apc_prefixes", None)
            or getattr(ctx.state, "apc_reused", False)
            or ctx.state.prompt_pos > 0
        ):
            return 0
        for seq_len, key in reversed(ctx.state.apc_prefixes):
            reused = self.prefix_cache.try_reuse(key, ctx.state.paged_seq_id, seq_len)
            if reused:
                ctx.state.prompt_pos = seq_len
                ctx.state.apc_reused = True
                if _PAGED_TRACE:
                    _PAGED_LOG.info(
                        "apc.reuse request=%s seq_id=%s tokens=%s",
                        getattr(ctx.state, "request_id", "unknown"),
                        ctx.state.paged_seq_id,
                        seq_len,
                    )
                return reused
        return 0

    def _record_apc_prefixes(self, ctx) -> None:
        if (
            self.prefix_cache is None
            or getattr(ctx.state, "apc_recorded", False)
            or getattr(ctx.state, "paged_seq_id", None) is None
            or not getattr(ctx.state, "apc_prefixes", None)
            or ctx.state.prompt_pos < ctx.state.prompt_len
        ):
            return
        self.prefix_cache.record_many(ctx.state.apc_prefixes, ctx.state.paged_seq_id)
        ctx.state.apc_recorded = True
        ctx.state.apc_prefixes = None
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "apc.record request=%s seq_id=%s prompt_len=%s reused=%s",
                getattr(ctx.state, "request_id", "unknown"),
                ctx.state.paged_seq_id,
                ctx.state.prompt_len,
                getattr(ctx.state, "apc_reused", False),
            )

    def prefill_slice_stats(self) -> Dict[str, float]:
        stats = dict(self._prefill_metrics)
        stats.setdefault("prefill_hybrid_dense_tokens", 0.0)
        stats.setdefault("prefill_hybrid_dense_ms", 0.0)
        stats.setdefault("paged_prefill_slice_s", 0.0)
        stats.setdefault("paged_prefill_model_s", 0.0)
        stats.setdefault("paged_prefill_commit_s", 0.0)
        stats.setdefault("paged_prefill_slice_count", 0.0)
        stats.setdefault("paged_prefill_first_slice_ms", 0.0)
        if self.use_array_prefill:
            stats.setdefault("array_prefill_graph_s", 0.0)
            stats.setdefault("array_prefill_writer_s", 0.0)
            stats.setdefault("array_prefill_pending_tokens", 0.0)
            stats.setdefault("array_prefill_pending_tokens_max", 0.0)
            stats.setdefault("array_prefill_attn_s", 0.0)
            stats.setdefault("array_prefill_mlp_s", 0.0)
            stats.setdefault("array_prefill_overlay_s", 0.0)
            stats.setdefault("array_prefill_overlay_wait_s", 0.0)
            stats.setdefault("array_prefill_overlay_wait_count", 0.0)
            stats.setdefault("array_prefill_waiting_sequences", 0.0)
            if (
                self._array_prefill_first_chunk_ms > 0.0
                and stats.get("array_prefill_first_chunk_ms", 0.0) <= 0.0
            ):
                stats["array_prefill_first_chunk_ms"] = (
                    self._array_prefill_first_chunk_ms
                )
                self._array_prefill_first_chunk_ms = 0.0
        self._prefill_metrics = {}
        return stats

    def _accumulate_array_prefill_metrics(
        self, stats: Optional[Dict[str, float]]
    ) -> None:
        if not stats or not isinstance(stats, dict):
            return
        for key in (
            "array_prefill_graph_s",
            "array_prefill_writer_s",
            "array_prefill_chunk_ms_total",
            "array_prefill_chunk_count",
            "array_prefill_first_chunk_ms",
            "array_prefill_pending_tokens",
            "array_prefill_pending_tokens_max",
            "array_prefill_attn_s",
            "array_prefill_mlp_s",
            "array_prefill_overlay_s",
            "array_prefill_overlay_wait_s",
            "array_prefill_overlay_wait_count",
            "array_prefill_waiting_sequences",
        ):
            value = float(stats.get(key, 0.0))
            if not value:
                continue
            if key == "array_prefill_first_chunk_ms":
                if self._array_prefill_first_chunk_ms <= 0.0:
                    self._array_prefill_first_chunk_ms = value
                continue
            self._prefill_metrics[key] = self._prefill_metrics.get(key, 0.0) + value

    def _consume_array_prefill_metrics(self) -> None:
        if self.prefill_array_runner is None:
            return
        try:
            metrics = self.prefill_array_runner.consume_metrics()
        except Exception:
            logging.debug(
                "SlotGenerator: consume_metrics failed",
                exc_info=True,
            )
            return
        if metrics:
            self._accumulate_array_prefill_metrics(metrics)

    def _maybe_compile_decode(self) -> None:
        if self._array_decode_flag:
            return
        if not _COMPILE_DECODE or self._compiled_decode is not None:
            return
        if not hasattr(mx, "compile"):
            return
        try:
            fn = lambda tokens, cache: self.model(tokens, cache=cache)
            self._compiled_decode = mx.compile(fn, shapeless=True)
            logging.info("SlotGenerator: enabled mx.compile for decode path")
        except Exception as exc:  # pragma: no cover - optional acceleration
            logging.warning("SlotGenerator: unable to compile decode path: %s", exc)
            self._compiled_decode = None

    def _call_model(
        self,
        inputs,
        cache,
        view,
        *,
        pending_state: Optional[ArrayDecodeRunner.PendingState] = None,
    ):
        if self.array_decode_runner is not None and isinstance(view, ArrayBatchView):
            return self.array_decode_runner.decode(
                inputs, view, pending_state=pending_state
            )
        if self._compiled_decode is not None:
            try:
                return self._compiled_decode(inputs, cache)
            except Exception as exc:  # pragma: no cover - optional acceleration
                logging.warning(
                    "SlotGenerator: compiled decode invocation failed, disabling: %s",
                    exc,
                )
                self._compiled_decode = None
        if view is not None:
            with batch_view_scope(view):
                return self.model(inputs, cache=cache)
        return self.model(inputs, cache=cache)

    def _run_with_backend(self, view, fn):
        if (
            self.use_array_decode
            and isinstance(view, ArrayBatchView)
            and self._paged_backend is not None
        ):
            self._paged_backend.set_batch_view(view)
            try:
                with use_attention_backend(self._paged_backend):
                    return fn()
            finally:
                self._paged_backend.clear_batch_view()
        if view is not None:
            with batch_view_scope(view):
                return fn()
        return fn()

    def _call_compiled_array_decode(self, inputs, cache, view):
        if not self._cache_is_array_tree(cache):
            return None
        key = self._compile_signature(view)
        if key is None:
            return None
        fn = self._compile_cache.get(key)
        if fn is None:
            try:
                fn = mx.compile(
                    lambda tokens, cache_param: self.model(tokens, cache=cache_param)
                )
            except Exception:  # pragma: no cover - compile fallback
                logging.warning(
                    "SlotGenerator: array decode compile failed", exc_info=True
                )
                return None
            self._compile_cache[key] = fn
            self._compile_cache_misses += 1
        else:
            self._compile_cache_hits += 1
        return self._run_with_backend(view, lambda: fn(inputs, cache))

    def _compile_signature(self, view):
        if not isinstance(view, ArrayBatchView):
            return None
        block_tables = getattr(view, "block_tables", None)
        if block_tables is None or not hasattr(block_tables, "shape"):
            return None
        batch = int(block_tables.shape[0])
        max_blocks = int(block_tables.shape[1])
        manager = self._paged_manager
        layers = getattr(manager, "num_layers", self._paged_num_layers or 0)
        kv_heads = getattr(manager, "num_kv_heads", 0)
        head_dim = getattr(manager, "head_dim", 0)
        block_size = getattr(manager, "block_size", self.prefill_chunk)
        dtype_obj = getattr(manager, "dtype", None)
        if dtype_obj is None and hasattr(block_tables, "dtype"):
            dtype_obj = block_tables.dtype
        dtype = str(getattr(dtype_obj, "name", dtype_obj))
        return (
            int(self._bucket(batch, getattr(self.slot_alloc, "capacity", batch or 1))),
            int(
                self._bucket(
                    max_blocks, getattr(manager, "max_blocks", max_blocks or 1)
                )
            ),
            int(layers or 0),
            int(kv_heads or 0),
            int(head_dim or 0),
            int(block_size or 0),
            dtype,
            self.kv_quant_mode,
        )

    @staticmethod
    def _bucket(value: int, limit: int) -> int:
        if value <= 0:
            return 1
        bucket = 1
        limit = max(limit, value)
        while bucket < value and bucket < limit:
            bucket <<= 1
        return bucket if bucket >= value else value

    def _cache_is_array_tree(self, cache) -> bool:
        import mlx.core as mx

        if cache is None:
            return True
        if isinstance(cache, (list, tuple)):
            return all(self._cache_is_array_tree(item) for item in cache)
        return isinstance(cache, mx.array)

    def compile_stats(self) -> Dict[str, float]:
        return {
            "compile_cache_hits": float(self._compile_cache_hits),
            "compile_cache_misses": float(self._compile_cache_misses),
            **(
                self.array_decode_runner.compile_stats()
                if self.array_decode_runner is not None
                else {}
            ),
        }

    # ------------------------------------------------------------------#
    # Decode helpers
    # ------------------------------------------------------------------#
    def decode_step(self, contexts: Sequence, decode_state: Optional[dict] = None):
        return self._decode_once(contexts, decode_state)

    def decode_chunk(
        self,
        contexts: Sequence,
        *,
        max_steps: int,
        decode_state: Optional[dict] = None,
        safe_bump: bool = True,
        emit_callback: Optional[
            callable
        ] = None,  # signature: (active, logits) -> Tuple[int, int]
    ) -> Dict[str, float]:
        total_iterations = 0
        total_model_s = 0.0
        total_kernel_s = 0.0
        total_s = 0.0
        total_vectorized = 0
        prior_active = sum(
            not getattr(ctx.state, "finished", False) for ctx in contexts
        )
        for _ in range(max(1, max_steps)):
            active = [
                ctx for ctx in contexts if not getattr(ctx.state, "finished", False)
            ]
            if not active:
                break
            logits = self._decode_once(active, decode_state, max_steps=max_steps)
            profile = dict(self._last_decode_profile)
            emitted = 0
            vectorized = 0
            if emit_callback is not None:
                emitted, vectorized = emit_callback(active, logits)
                total_vectorized += vectorized
            total_model_s += profile.get("model_s", 0.0)
            total_kernel_s += profile.get("kernel_s", 0.0)
            total_s += profile.get("total_s", profile.get("model_s", 0.0))
            if emitted:
                total_iterations += 1
            if emit_callback is None or emitted == 0:
                break
            if decode_state is not None:
                self.update_decode_state(decode_state, contexts, safe_bump=safe_bump)
            current_active = sum(
                not getattr(ctx.state, "finished", False) for ctx in contexts
            )
            if current_active < prior_active:
                break
            prior_active = current_active
        if (
            decode_state is not None
            and self.array_decode_runner is not None
            and decode_state.get("pending") is not None
        ):
            self.array_decode_runner.flush_pending(decode_state["pending"])
            decode_state["pending"] = None
        if (
            self.use_array_prefill
            and self.prefill_array_runner is not None
            and not self.prefill_array_runner.has_active_overlays()
        ):
            self.prefill_array_runner.flush_pending()
            self._consume_array_prefill_metrics()
        return {
            "iterations": total_iterations,
            "model_s": total_model_s,
            "kernel_s": total_kernel_s,
            "total_s": total_s,
            "vectorized": total_vectorized,
        }

    def _decode_once(
        self,
        contexts: Sequence,
        decode_state: Optional[dict] = None,
        *,
        max_steps: int = 1,
    ):
        active = [ctx for ctx in contexts if not getattr(ctx.state, "finished", False)]
        if not active:
            self._last_decode_profile = {
                "batch_size": 0,
                "reuse": False,
                "model_s": 0.0,
                "total_s": 0.0,
            }
            return mx.zeros((0, 0), dtype=mx.float32)

        total_start = time.perf_counter()
        for ctx in active:
            if not getattr(ctx, "slot_initialized", False):
                if not self.on_admit(ctx):
                    continue
            self._sync_prompt_cache(ctx)

        layer_count = len(active[0].prompt_cache)
        use_paged_backend = self.paged_cache is not None
        if use_paged_backend:
            batch_caches = self._ensure_paged_batch_layers(layer_count)
        else:
            if self._slab is None:
                self._slab = SlotKVSlab(
                    self.slot_alloc.capacity, initial_capacity=self.prefill_chunk
                )
            if len(self._batched_cache_layers) != layer_count:
                self._batched_cache_layers = [
                    SlotBatchCache(self._slab, layer_idx)
                    for layer_idx in range(layer_count)
                ]
            batch_caches = self._batched_cache_layers
        for layer_cache in batch_caches:
            layer_cache.bind(active)

        tokens = []
        for ctx in active:
            token = getattr(ctx, "last_token_id", None)
            if token is None:
                token = getattr(self.tokenizer, "bos_token_id", 0)
            tokens.append(int(token))
        inputs = mx.array(tokens, dtype=mx.int32)[:, None]

        model_start = time.perf_counter()
        view = None
        if use_paged_backend:
            view = self._resolve_paged_view(active, decode_state, max_steps=max_steps)

        pending_state = None
        if (
            use_paged_backend
            and self.use_array_decode
            and self.array_decode_runner is not None
            and decode_state is not None
            and view is not None
            and max_steps > 1
        ):
            pending_state = decode_state.get("pending")
            if pending_state is None or pending_state.view is not view:
                pending_state = self.array_decode_runner.new_pending_state(
                    view, capacity=max_steps
                )
                decode_state["pending"] = pending_state

        logits = self._call_model(
            inputs, batch_caches, view, pending_state=pending_state
        )
        overlay_tokens = (
            int(getattr(view, "_prefill_overlay_tokens", 0) or 0)
            if isinstance(view, ArrayBatchView)
            else 0
        )
        if (
            self.prefill_array_runner is not None
            and isinstance(view, ArrayBatchView)
            and view.seq_ids
            and overlay_tokens > 0
        ):
            try:
                overlay_seq_ids = tuple(
                    getattr(view, "_prefill_overlay_seq_ids", tuple())
                )
                if overlay_seq_ids:
                    self.prefill_array_runner.advance_and_flush(
                        overlay_seq_ids, overlay_tokens
                    )
            except Exception:
                logging.debug("SlotGenerator: advance_and_flush failed", exc_info=True)
            finally:
                self._consume_array_prefill_metrics()
            overlay_tokens = 0
        if isinstance(view, ArrayBatchView):
            setattr(view, "_prefill_overlay_tokens", overlay_tokens)
            if overlay_tokens == 0:
                setattr(view, "_prefill_overlay_seq_ids", tuple())
            view.prefill_overlays = None
            view.prefill_overlay_rows = ()
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        kernel_ms = None
        if hasattr(mx.fast, "_paged_attention_last_time_ms"):
            try:
                kernel_ms = float(mx.fast._paged_attention_last_time_ms())
            except Exception:  # pragma: no cover - debug helper
                kernel_ms = None
        model_elapsed = time.perf_counter() - model_start
        total_elapsed = time.perf_counter() - total_start
        self._last_decode_profile = {
            "batch_size": len(active),
            "reuse": True,
            "model_s": model_elapsed,
            "total_s": total_elapsed,
        }
        if kernel_ms is not None:
            self._last_decode_profile["kernel_s"] = kernel_ms / 1000.0
        logging.debug("SlotGenerator.decode profile %s", self._last_decode_profile)
        if _BENCH_TRACE:
            backend = "paged" if self.paged_cache is not None else "dense"
            _BENCH_LOG.info(
                "slot.decode batch=%s paged_view=%s backend=%s model_ms=%.3f total_ms=%.3f",
                len(active),
                view is not None,
                backend,
                model_elapsed * 1000.0,
                total_elapsed * 1000.0,
            )
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "slot.decode.profile batch=%s paged_view=%s model_ms=%.3f total_ms=%.3f kernel_ms=%s",
                len(active),
                view is not None,
                model_elapsed * 1000.0,
                total_elapsed * 1000.0,
                None if kernel_ms is None else round(kernel_ms, 3),
            )

        return logits

    def _build_new_view(self, active: Sequence[SequenceContext]):
        seq_ids: List[int] = []
        for ctx in active:
            seq_id = getattr(ctx.state, "paged_seq_id", None)
            if seq_id is None:
                seq_ids = []
                break
            seq_ids.append(seq_id)
        if not seq_ids:
            if _PAGED_TRACE:
                missing = [
                    getattr(ctx.state, "request_id", "unknown")
                    for ctx in active
                    if getattr(ctx.state, "paged_seq_id", None) is None
                ]
                if missing:
                    _PAGED_LOG.info("slot.decode.no_view missing_seq_ids=%s", missing)
            return None
        view = self.paged_cache.make_batch_view(
            seq_ids, kv_head_mapping=self.kv_head_mapping
        )
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "slot.decode.view batch=%s seq_ids=%s context=%s tables_shape=%s",
                len(seq_ids),
                _seq_preview(seq_ids),
                _array_preview(view.context_lens),
                tuple(view.block_tables.shape),
            )
        return view

    def _resolve_paged_view(
        self,
        active: Sequence[SequenceContext],
        decode_state: Optional[dict],
        *,
        max_steps: int = 1,
    ):
        if decode_state is None:
            return self._maybe_array_view(
                self._build_new_view(active), max_steps=max_steps
            )
        seq_ids = tuple(getattr(ctx.state, "paged_seq_id", None) for ctx in active)
        if not seq_ids or any(seq_id is None for seq_id in seq_ids):
            decode_state["view"] = None
            decode_state["seq_ids"] = ()
            if _PAGED_TRACE:
                missing = [
                    getattr(ctx.state, "request_id", "unknown")
                    for ctx in active
                    if getattr(ctx.state, "paged_seq_id", None) is None
                ]
                if missing:
                    _PAGED_LOG.info("slot.decode.no_view missing_seq_ids=%s", missing)
            return None
        if decode_state.get("seq_ids") != seq_ids or decode_state.get("view") is None:
            view = self.paged_cache.make_batch_view(
                seq_ids, kv_head_mapping=self.kv_head_mapping
            )
            decode_state["view"] = self._maybe_array_view(view, max_steps=max_steps)
            decode_state["seq_ids"] = seq_ids
            if _PAGED_TRACE:
                _PAGED_LOG.info(
                    "slot.decode.view batch=%s seq_ids=%s context=%s tables_shape=%s",
                    len(seq_ids),
                    _seq_preview(seq_ids),
                    _array_preview(view.context_lens),
                    tuple(view.block_tables.shape),
                )
            return decode_state["view"]
        return self._maybe_array_view(decode_state.get("view"), max_steps=max_steps)

    def _maybe_array_view(self, view, *, max_steps: int = 1):
        if view is None or self._arrays_provider is None:
            return view
        if isinstance(view, ArrayBatchView):
            self._attach_array_overlays(view, max_steps=max_steps)
            return view
        seq_ids = getattr(view, "seq_ids", None)
        if not seq_ids:
            return view
        try:
            array_view = self._arrays_provider.build_view(
                view, seq_ids, decode_steps=max_steps
            )
            if self.prefill_array_runner is not None:
                self._attach_array_overlays(array_view, max_steps=max_steps)
            return array_view
        except Exception:
            logging.debug("SlotGenerator: array view build failed", exc_info=True)
            return view

    def _attach_array_overlays(self, view, *, max_steps: int = 1) -> None:
        if (
            self.prefill_array_runner is None
            or not isinstance(view, ArrayBatchView)
            or not view.seq_ids
        ):
            view.prefill_overlays = None
            view.prefill_overlay_rows = ()
            setattr(view, "_prefill_overlay_tokens", 0)
            setattr(view, "_prefill_overlay_seq_ids", ())
            return
        steps = max(1, int(max_steps or 1))
        ramp = getattr(self, "prefill_ramp_chunk", None)
        if ramp:
            steps = max(1, min(steps, int(ramp)))
        if steps <= 0:
            view.prefill_overlays = None
            view.prefill_overlay_rows = ()
            setattr(view, "_prefill_overlay_tokens", 0)
            setattr(view, "_prefill_overlay_seq_ids", ())
            return
        overlay_seq_ids: Tuple[int, ...] = ()
        try:
            overlay_seq_ids = self.prefill_array_runner.overlay_seq_ids(view.seq_ids)
        except AttributeError:
            overlay_seq_ids = tuple(int(seq_id) for seq_id in view.seq_ids)
        if not overlay_seq_ids:
            view.prefill_overlays = None
            view.prefill_overlay_rows = ()
            setattr(view, "_prefill_overlay_tokens", 0)
            setattr(view, "_prefill_overlay_seq_ids", ())
            return
        overlay_batch = self.prefill_array_runner.collect_overlays(
            overlay_seq_ids, max(1, steps)
        )
        if overlay_batch is None or overlay_batch.tokens <= 0:
            view.prefill_overlays = None
            view.prefill_overlay_rows = ()
            setattr(view, "_prefill_overlay_tokens", 0)
            setattr(view, "_prefill_overlay_seq_ids", ())
            return
        row_map = {int(seq_id): idx for idx, seq_id in enumerate(view.seq_ids)}
        overlay_rows = tuple(
            row_map[seq_id] for seq_id in overlay_seq_ids if seq_id in row_map
        )
        if len(overlay_rows) != len(overlay_seq_ids):
            view.prefill_overlays = None
            view.prefill_overlay_rows = ()
            setattr(view, "_prefill_overlay_tokens", 0)
            setattr(view, "_prefill_overlay_seq_ids", ())
            return
        view.prefill_overlays = overlay_batch
        view.prefill_overlay_rows = overlay_rows
        setattr(view, "_prefill_overlay_tokens", overlay_batch.tokens)
        setattr(view, "_prefill_overlay_seq_ids", tuple(overlay_seq_ids))

    def update_decode_state(
        self,
        decode_state: Optional[dict],
        active: Sequence[SequenceContext],
        *,
        safe_bump: bool,
    ) -> None:
        if decode_state is None:
            return
        view = decode_state.get("view")
        if view is None:
            return
        pending_state = decode_state.get("pending")

        def _flush_pending() -> None:
            nonlocal pending_state
            if pending_state is not None and self.array_decode_runner is not None:
                self.array_decode_runner.flush_pending(pending_state)
                decode_state["pending"] = None
                pending_state = None

        seq_ids = decode_state.get("seq_ids", ())
        current_ids = tuple(getattr(ctx.state, "paged_seq_id", None) for ctx in active)
        if seq_ids != current_ids or not safe_bump:
            _flush_pending()
            decode_state["view"] = None
            decode_state["seq_ids"] = ()
            return
        if not current_ids:
            _flush_pending()
            decode_state["view"] = None
            decode_state["seq_ids"] = ()
            return
        deltas: List[int] = []
        for ctx in active:
            deltas.append(0 if getattr(ctx.state, "finished", False) else 1)
        if not deltas:
            decode_state["view"] = None
            decode_state["seq_ids"] = ()
            return
        if self.paged_cache.can_bump_view(view, deltas):
            view.bump_context(deltas)
        else:
            _flush_pending()
            decode_state["view"] = None
            decode_state["seq_ids"] = ()

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _sync_prompt_cache(self, ctx) -> None:
        if (
            self.paged_cache is not None
            and getattr(ctx.state, "paged_seq_id", None) is not None
        ):
            return
        if getattr(ctx.state, "slot_id", None) is None:
            return
        if self._slab is None:
            self._slab = SlotKVSlab(
                self.slot_alloc.capacity, initial_capacity=self.prefill_chunk
            )
        caches = getattr(ctx, "prompt_cache", None)
        if caches is None:
            return
        if not hasattr(ctx, "_kv_lengths"):
            ctx._kv_lengths = [0] * len(caches)
        slab = self._slab
        slot = ctx.state.slot_id
        for layer_idx, cache in enumerate(caches):
            state = cache.state
            if isinstance(state, tuple):
                keys, values = state[:2]
            else:
                keys, values = state
            if keys is None:
                continue
            total = int(getattr(cache, "offset", 0))
            prev = ctx._kv_lengths[layer_idx]
            if total <= prev:
                continue
            slab.ensure_layer(layer_idx, keys, values)
            slab.append_from_cache(layer_idx, slot, cache, prev, total)
            ctx._kv_lengths[layer_idx] = total

    def _ensure_paged_batch_layers(self, layer_count: int) -> list[PagedBatchCache]:
        if len(self._paged_batch_layers) != layer_count:
            self._paged_batch_layers = [
                PagedBatchCache(idx) for idx in range(layer_count)
            ]
        return self._paged_batch_layers

    def _refresh_paged_lengths(self, ctx) -> None:
        caches = getattr(ctx, "prompt_cache", None)
        if not isinstance(caches, list):
            return
        if not hasattr(ctx, "_kv_lengths"):
            ctx._kv_lengths = [0] * len(caches)
        for idx, cache in enumerate(caches):
            offset = getattr(cache, "offset", None)
            if offset is not None:
                ctx._kv_lengths[idx] = int(offset)

    def _ensure_prompt_cache(self, ctx) -> None:
        prompt_tokens = getattr(ctx, "prompt_tokens", [])
        cached = getattr(ctx, "prompt_ids_np", None)
        if cached is None or len(cached) != len(prompt_tokens):
            ctx.prompt_ids_np = np.array(prompt_tokens, dtype=np.int32)
        if getattr(ctx, "prompt_cache", None) is None:
            ctx.prompt_cache = make_prompt_cache(self.model)


__all__ = ["SlotBatchCache", "PagedBatchCache", "SlotGenerator"]
