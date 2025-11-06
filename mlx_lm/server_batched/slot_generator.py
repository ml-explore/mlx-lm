# ABOUTME: Coordinates slot-indexed prompt prefill and decode loops.
# ABOUTME: Provides vectorized model calls while managing slot assignments.

from __future__ import annotations

from typing import List, Optional, Sequence

import logging
import time

import numpy as np

import mlx.core as mx

from ..models.cache import make_prompt_cache


def _mx_array(data, *, dtype=None):
    if dtype is None and hasattr(mx, "int32"):
        dtype = mx.int32
    try:
        if dtype is not None:
            return mx.array(data, dtype=dtype)
        return mx.array(data)
    except TypeError:
        return mx.array(data)

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
            empty = mx.zeros((len(slot_ids), self.n_kv_heads, 0, self.head_dim), dtype=self.dtype)
            return empty, empty
        batch = len(slot_ids)
        keys_full = mx.zeros(
            (batch, max_len, self.n_kv_heads, self.head_dim), dtype=self.dtype
        )
        values_full = mx.zeros((batch, max_len, self.n_kv_heads, self.head_dim), dtype=self.dtype)
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

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.layers: List[Optional[SlotLayerSlab]] = []

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
        initial_capacity = int(keys.shape[2])
        self.layers[layer_idx] = SlotLayerSlab(
            self.max_slots,
            n_kv_heads,
            head_dim,
            dtype,
            initial_capacity=initial_capacity,
        )

    def append_from_cache(self, layer_idx: int, slot_id: int, cache, start: int, end: int) -> None:
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

    def append_batch(self, layer_idx: int, slot_ids: Sequence[int], keys, values) -> None:
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
            offsets = [lp + length for lp, length in zip(left_padding, self._scratch_lengths)]
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
        keys, values = (state[:2] if isinstance(state, tuple) else (state, None))
        self.slab.ensure_layer(self.layer_idx, keys, values)
        layer = self.slab.layers[self.layer_idx]

        new_lengths = [ctx._kv_lengths[self.layer_idx] for ctx in contexts]
        capacity_target = layer.capacity if layer is not None else max(max(new_lengths, default=0), 1)
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
        if self._scratch_keys is None or self._scratch_values is None or not self.slot_ids:
            empty = mx.zeros((0, 0, 0, 0), dtype=mx.float32)
            offsets = mx.zeros((0,), dtype=mx.int32)
            return empty, empty, offsets, self.left_padding
        start = self._capacity - self._max_len
        gathered_k = self._scratch_keys[:, start:self._capacity]
        gathered_v = self._scratch_values[:, start:self._capacity]
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
                    prev_len = self._scratch_lengths[idx] if idx < len(self._scratch_lengths) else 0
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


class SlotGenerator:
    """Slot-indexed generator that batches model forwards for decode."""

    def __init__(self, model, tokenizer, slot_alloc, *, prefill_chunk: int = 1024):
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.slot_alloc = slot_alloc
        self.prefill_chunk = prefill_chunk
        self._slab: Optional[SlotKVSlab] = None
        self._batched_cache_layers: list[SlotBatchCache] = []
        self._last_decode_profile = {}

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
        return True

    def on_release(self, ctx) -> None:
        slot_id = getattr(ctx.state, "slot_id", None)
        if slot_id is not None:
            self.slot_alloc.release(slot_id)
            if self._slab is not None:
                self._slab.reset_slot(slot_id)
            ctx.state.slot_id = None
        ctx.slot_initialized = False

    # ------------------------------------------------------------------#
    # Prefill helpers
    # ------------------------------------------------------------------#
    def prefill_step(self, contexts: Sequence) -> None:
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
        remaining = ctx.state.remaining_prompt_tokens
        if remaining <= 0 or max_tokens <= 0:
            return 0
        take = min(max_tokens, remaining)
        start = ctx.state.prompt_pos
        end = start + take
        span = ctx.prompt_ids_np[start:end]
        if span.size == 0:
            ctx.state.prompt_pos = end
            return 0
        inputs = np.expand_dims(span, axis=0)
        self.model(_mx_array(inputs), cache=ctx.prompt_cache)
        ctx.state.prompt_pos = end
        ctx.last_token_id = int(span[-1])
        self._sync_prompt_cache(ctx)
        return take

    # ------------------------------------------------------------------#
    # Decode helpers
    # ------------------------------------------------------------------#
    def decode_step(self, contexts: Sequence):
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
        if self._slab is None:
            self._slab = SlotKVSlab(self.slot_alloc.capacity)
        if len(self._batched_cache_layers) != layer_count:
            self._batched_cache_layers = [
                SlotBatchCache(self._slab, layer_idx) for layer_idx in range(layer_count)
            ]
        for layer_cache in self._batched_cache_layers:
            layer_cache.bind(active)
        batch_caches = self._batched_cache_layers

        tokens = []
        for ctx in active:
            token = getattr(ctx, "last_token_id", None)
            if token is None:
                token = getattr(self.tokenizer, "bos_token_id", 0)
            tokens.append(int(token))
        inputs = mx.array(tokens, dtype=mx.int32)[:, None]

        model_start = time.perf_counter()
        logits = self.model(inputs, cache=batch_caches)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        mx.eval(logits)
        model_elapsed = time.perf_counter() - model_start

        total_elapsed = time.perf_counter() - total_start
        self._last_decode_profile = {
            "batch_size": len(active),
            "reuse": True,
            "model_s": model_elapsed,
            "total_s": total_elapsed,
        }
        logging.debug(
            "SlotGenerator.decode profile %s", self._last_decode_profile
        )

        return logits

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _sync_prompt_cache(self, ctx) -> None:
        if getattr(ctx.state, "slot_id", None) is None:
            return
        if self._slab is None:
            self._slab = SlotKVSlab(self.slot_alloc.capacity)
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

    def _ensure_prompt_cache(self, ctx) -> None:
        prompt_tokens = getattr(ctx, "prompt_tokens", [])
        cached = getattr(ctx, "prompt_ids_np", None)
        if cached is None or len(cached) != len(prompt_tokens):
            ctx.prompt_ids_np = np.array(prompt_tokens, dtype=np.int32)
        if getattr(ctx, "prompt_cache", None) is None:
            ctx.prompt_cache = make_prompt_cache(self.model)



__all__ = ["SlotBatchCache", "SlotGenerator"]
