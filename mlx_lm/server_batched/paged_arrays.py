# ABOUTME: Produces array-backed paged-attention views for decode.
# ABOUTME: Provides lightweight batch views detached from the KV manager.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from .paged_slot_kv_cache import ViewSignature


@dataclass
class PagedDecodeArrays:
    seq_ids: Tuple[int, ...]
    block_tables: mx.array  # [B, max_blocks]
    context_lens: mx.array  # [B]
    kv_head_mapping: Optional[mx.array]
    k_cache: mx.array  # [L, KvH, max_blocks, block_size, head_dim]
    v_cache: mx.array
    quant_kwargs: List[Dict[str, object]]
    write_block_ids: mx.array  # [B]
    write_token_offsets: mx.array  # [B]
    prefill_base_lens: Optional[mx.array]
    signature: Optional[ViewSignature]
    active_rows: int


@dataclass
class PrefillOverlayView:
    rows: Tuple[int, ...]
    k: mx.array
    v: mx.array
    base_lens: mx.array
    tokens: int


class ArrayBatchView:
    """Batch view that serves paged attention arguments from prebuilt arrays."""

    def __init__(self, arrays: PagedDecodeArrays, *, prefill_overlays=None):
        self.seq_ids = arrays.seq_ids
        self.block_tables = arrays.block_tables
        self.context_lens = arrays.context_lens
        self.kv_head_mapping = arrays.kv_head_mapping
        self._k_cache = arrays.k_cache
        self._v_cache = arrays.v_cache
        self._quant_kwargs = arrays.quant_kwargs
        self.write_block_ids = arrays.write_block_ids
        self.write_token_offsets = arrays.write_token_offsets
        self.prefill_base_lens = arrays.prefill_base_lens
        self.signature = arrays.signature
        self.active_rows = arrays.active_rows
        self.prefill_overlays = prefill_overlays
        self._prefill_overlay_tokens = 0
        self.prefill_overlay_rows: Tuple[int, ...] = ()
        self._prefill_overlay_seq_ids: Tuple[int, ...] = ()

    def args_for_layer(self, layer_idx: int):
        return (
            self._k_cache[layer_idx],
            self._v_cache[layer_idx],
            self.block_tables,
            self.context_lens,
            layer_idx,
            self.kv_head_mapping,
            dict(
                self._quant_kwargs[layer_idx]
                if layer_idx < len(self._quant_kwargs)
                else {}
            ),
        )

    def overlay_for_layer(self, layer_idx: int):
        if self.prefill_overlays is None:
            return None
        rows = tuple(getattr(self, "prefill_overlay_rows", ()))
        if not rows:
            return None
        k_tensor, v_tensor, base_lens, tokens = self.prefill_overlays.layer_tensors(
            layer_idx
        )
        return PrefillOverlayView(
            rows=rows,
            k=k_tensor,
            v=v_tensor,
            base_lens=base_lens,
            tokens=int(tokens),
        )

    def bump_context(self, delta: Sequence[int]) -> None:
        if not delta:
            return
        delta_arr = mx.array(delta, dtype=self.context_lens.dtype)
        self.context_lens = self.context_lens + delta_arr

    @property
    def size(self) -> int:
        return len(self.seq_ids)


class PagedArraysProvider:
    """Builds array-backed batch views detached from the KV manager."""

    def __init__(self, manager, *, num_layers: int, block_size: int):
        self.manager = manager
        self.num_layers = num_layers
        self.block_size = block_size
        self._quant_helper = getattr(manager, "_quant_attention_kwargs", None)

    def build_view(
        self,
        view,
        seq_ids: Sequence[int],
        *,
        decode_steps: int = 1,
    ) -> ArrayBatchView:
        seq_ids = tuple(seq_ids)
        self.manager.ensure_decode_capacity(seq_ids, tokens=decode_steps)
        context = mx.array(view.context_lens, dtype=mx.int32)
        block_tables = mx.array(view.block_tables, dtype=mx.int32)
        write_blocks, write_offsets = self._write_targets(seq_ids)
        quant_kwargs: List[Dict[str, object]] = []
        for layer_idx in range(self.num_layers):
            layer_kwargs = {}
            if self._quant_helper is not None:
                layer_kwargs = dict(self._quant_helper(layer_idx) or {})
            quant_kwargs.append(layer_kwargs)
        arrays = PagedDecodeArrays(
            seq_ids=tuple(seq_ids),
            block_tables=block_tables,
            context_lens=context,
            kv_head_mapping=view.kv_head_mapping,
            k_cache=self.manager.k,
            v_cache=self.manager.v,
            quant_kwargs=quant_kwargs,
            write_block_ids=write_blocks,
            write_token_offsets=write_offsets,
            prefill_base_lens=getattr(view, "prefill_base_lens", None),
            signature=getattr(view, "signature", None),
            active_rows=len(seq_ids),
        )
        return ArrayBatchView(arrays)

    def _write_targets(self, seq_ids: Sequence[int]):
        block_ids, offsets = self.manager.decode_write_targets(seq_ids)
        return mx.array(block_ids, dtype=mx.int32), mx.array(offsets, dtype=mx.int32)


__all__ = [
    "ArrayBatchView",
    "PagedArraysProvider",
    "PagedDecodeArrays",
    "PrefillOverlayView",
]
