# ABOUTME: Coordinates slot-indexed prompt prefill and decode loops.
# ABOUTME: Provides vectorized model calls while managing slot assignments.

from __future__ import annotations

from typing import Sequence

import numpy as np

import mlx.core as mx


def _mx_array(data, *, dtype=None):
    if dtype is None and hasattr(mx, "int32"):
        dtype = mx.int32
    try:
        if dtype is not None:
            return mx.array(data, dtype=dtype)
        return mx.array(data)
    except TypeError:
        return mx.array(data)


class SlotGenerator:
    """Slot-indexed generator that batches model forwards for decode."""

    def __init__(self, model, tokenizer, slot_alloc, *, prefill_chunk: int = 1024):
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.slot_alloc = slot_alloc
        self.prefill_chunk = prefill_chunk

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
        self.model(_mx_array(inputs))
        ctx.state.prompt_pos = end
        ctx.last_token_id = int(span[-1])
        return take

    # ------------------------------------------------------------------#
    # Decode helpers
    # ------------------------------------------------------------------#
    def decode_step(self, contexts: Sequence):
        active = [ctx for ctx in contexts if not getattr(ctx.state, "finished", False)]
        if not active:
            return mx.zeros((0, 0), dtype=mx.float32)
        for ctx in active:
            if not getattr(ctx, "slot_initialized", False):
                if not self.on_admit(ctx):
                    continue
        last_ids = []
        for ctx in active:
            token = getattr(ctx, "last_token_id", None)
            if token is None:
                token = getattr(self.tokenizer, "bos_token_id", 0)
            last_ids.append(int(token))
        batch_np = np.array(last_ids, dtype=np.int32).reshape(-1, 1)
        logits = self.model(_mx_array(batch_np))
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        return logits

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _ensure_prompt_cache(self, ctx) -> None:
        prompt_tokens = getattr(ctx, "prompt_tokens", [])
        cached = getattr(ctx, "prompt_ids_np", None)
        if cached is None or len(cached) != len(prompt_tokens):
            ctx.prompt_ids_np = np.array(prompt_tokens, dtype=np.int32)


__all__ = ["SlotGenerator"]
