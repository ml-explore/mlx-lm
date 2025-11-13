# ABOUTME: Provides server-scoped attention backend implementations.
# ABOUTME: Supplies paged-attention backend for use with SlotGenerator.

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from ..models.base import AttentionBackend
from .paged_slot_kv_cache import PagedBatchView, PagedSlotKVCache


class PagedAttentionBackend(AttentionBackend):
    """Attention backend that invokes mx.fast.paged_attention."""

    def __init__(
        self,
        paged_cache: PagedSlotKVCache,
        *,
        kv_head_mapping: Optional[mx.array] = None,
    ):
        self.paged_cache = paged_cache
        self.kv_head_mapping = kv_head_mapping
        self._view: Optional[PagedBatchView] = None

    def set_batch_view(self, view: PagedBatchView) -> None:
        self._view = view

    def clear_batch_view(self) -> None:
        self._view = None

    def run(
        self,
        queries: mx.array,
        *,
        keys: mx.array,
        values: mx.array,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array],
    ) -> mx.array:
        view = self._view
        if view is None:
            raise RuntimeError("PagedAttentionBackend requires an active batch view")
        (
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            layer_idx,
            kv_head_map,
            quant_kwargs,
        ) = view.args_for_layer(cache.layer_idx if hasattr(cache, "layer_idx") else 0)
        kv_head_mapping = kv_head_map or self.kv_head_mapping
        call_kwargs = dict(quant_kwargs)
        call_kwargs["layer_idx"] = layer_idx
        call_kwargs["kv_head_mapping"] = kv_head_mapping
        call_kwargs["scale"] = scale
        if mask is not None:
            call_kwargs["mask"] = mask
        if sinks is not None:
            call_kwargs["sinks"] = sinks
        return mx.fast.paged_attention(
            queries,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            **call_kwargs,
        )


__all__ = ["PagedAttentionBackend"]
