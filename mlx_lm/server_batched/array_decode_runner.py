# ABOUTME: Coordinates array-only decode execution for paged attention models.
# ABOUTME: Provides eager decode runner that avoids Python cache objects.

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mlx.core as mx

from .graph_decode import LlamaArrayGraph
from .paged_arrays import ArrayBatchView, PagedDecodeArrays, PrefillOverlayView

_LOG = logging.getLogger("mlx_lm.array_decode")
_ARRAY_DEBUG = os.environ.get("MLXLM_ARRAY_DEBUG", "").lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}


class ArrayDecodeRunner:
    """Runs decode steps using array-backed views."""

    SUPPORTED_ENGINES = frozenset({"paged-arrays", "paged-arrays+compile"})

    def __init__(self, model, manager, *, decode_engine: str):
        self.manager = manager
        self.decode_engine = decode_engine
        self.graph = self._build_graph(model)
        self._phase1_layers: dict[int, dict[tuple, callable]] = {}
        self._phase2_layers: dict[int, dict[tuple, callable]] = {}
        self._phase1_compile_hits = 0
        self._phase1_compile_misses = 0
        self._phase2_compile_hits = 0
        self._phase2_compile_misses = 0
        self._phase1_duration_s = 0.0
        self._attn_duration_s = 0.0
        self._outproj_duration_s = 0.0
        self._mlp_duration_s = 0.0
        self._writer_duration_s = 0.0
        self._phase2_compiled_duration_s = 0.0
        self._logged_q_shape = False

    @dataclass
    class WriterBatch:
        layer_idx: int
        k_cache: mx.array
        v_cache: mx.array
        seq_ids: Sequence[int]
        block_tables: mx.array
        context_lens: mx.array
        k_batch: mx.array
        v_batch: mx.array
        quant_kwargs: dict
        active_rows: int

    @dataclass
    class PendingState:
        view: ArrayBatchView
        capacity: int
        active_rows: int
        layer_count: int
        layer_batches: list[list["ArrayDecodeRunner.WriterBatch"]]
        overlay_k_buffers: list[Optional[mx.array]]
        overlay_v_buffers: list[Optional[mx.array]]
        overlay_lengths: list[int]
        pending_steps: int = 0

        def record_writer_batch(
            self,
            layer_idx: int,
            writer_batch: "ArrayDecodeRunner.WriterBatch",
        ) -> None:
            if self.pending_steps >= self.capacity:
                raise RuntimeError("pending state exceeded capacity")
            self.layer_batches[layer_idx].append(writer_batch)

        def reset(self) -> None:
            self.layer_batches = [[] for _ in range(self.layer_count)]
            self.overlay_k_buffers = [None for _ in range(self.layer_count)]
            self.overlay_v_buffers = [None for _ in range(self.layer_count)]
            self.overlay_lengths = [0 for _ in range(self.layer_count)]
            self.pending_steps = 0

    def _build_graph(self, model) -> LlamaArrayGraph:
        try:
            return LlamaArrayGraph(model)
        except ValueError as exc:
            raise RuntimeError(
                "Array decode runner currently supports only LLaMA models"
            ) from exc

    def decode(
        self,
        tokens: mx.array,
        view: ArrayBatchView,
        *,
        pending_state: Optional["ArrayDecodeRunner.PendingState"] = None,
    ) -> mx.array:
        if not isinstance(view, ArrayBatchView):
            raise TypeError("array decode runner requires an ArrayBatchView")
        if tokens.ndim == 2 and tokens.shape[1] == 1:
            token_ids = mx.reshape(tokens, (tokens.shape[0],))
        elif tokens.ndim == 1:
            token_ids = tokens
        else:
            token_ids = tokens[:, -1]
        if (
            pending_state is not None
            and pending_state.pending_steps >= pending_state.capacity
        ):
            raise RuntimeError("pending capacity exceeded for array decode runner")
        hidden = self.graph.embed(token_ids)
        if hidden.ndim == 2:
            hidden = mx.expand_dims(hidden, axis=1)
        writer_batches: list[Optional[ArrayDecodeRunner.WriterBatch]] = []
        for layer_idx, layer in enumerate(self.graph.layers):
            hidden, writer_batch = self._run_layer(
                layer_idx, layer, hidden, view, pending_state=pending_state
            )
            if pending_state is None:
                writer_batches.append(writer_batch)
        if pending_state is None:
            self._flush_writer_batches(writer_batches)
            if view.seq_ids:
                self.manager.bump_decode_lengths(view.seq_ids, 1)
        else:
            pending_state.pending_steps += 1
        logits = self.graph.logits(hidden)
        return logits[:, -1, :]

    def new_pending_state(
        self, view: ArrayBatchView, *, capacity: int
    ) -> "ArrayDecodeRunner.PendingState":
        layer_count = len(self.graph.layers)
        return ArrayDecodeRunner.PendingState(
            view=view,
            capacity=max(1, capacity),
            active_rows=view.active_rows,
            layer_count=layer_count,
            layer_batches=[[] for _ in range(layer_count)],
            overlay_k_buffers=[None for _ in range(layer_count)],
            overlay_v_buffers=[None for _ in range(layer_count)],
            overlay_lengths=[0 for _ in range(layer_count)],
        )

    def _pad_overlay(self, tensor: mx.array, capacity: int) -> mx.array:
        if tensor.shape[0] >= capacity:
            return tensor if tensor.shape[0] == capacity else tensor[:capacity]
        pad_shape = (capacity - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad = mx.zeros(pad_shape, dtype=tensor.dtype)
        return mx.concatenate([tensor, pad], axis=0)

    def _ensure_overlay_buffers(
        self,
        pending_state: "ArrayDecodeRunner.PendingState",
        layer_idx: int,
        sample_k: mx.array,
        sample_v: mx.array,
    ) -> tuple[mx.array, mx.array]:
        needed_shape = (
            pending_state.capacity,
            sample_k.shape[0],
            sample_k.shape[1],
            sample_k.shape[2],
        )
        buf_k = pending_state.overlay_k_buffers[layer_idx]
        buf_v = pending_state.overlay_v_buffers[layer_idx]
        if (
            buf_k is None
            or buf_k.shape != needed_shape
            or buf_k.dtype != sample_k.dtype
        ):
            buf_k = mx.zeros(needed_shape, dtype=sample_k.dtype)
            pending_state.overlay_k_buffers[layer_idx] = buf_k
            pending_state.overlay_lengths[layer_idx] = 0
        if (
            buf_v is None
            or buf_v.shape != needed_shape
            or buf_v.dtype != sample_v.dtype
        ):
            buf_v = mx.zeros(needed_shape, dtype=sample_v.dtype)
            pending_state.overlay_v_buffers[layer_idx] = buf_v
            pending_state.overlay_lengths[layer_idx] = 0
        return buf_k, buf_v

    def flush_pending(
        self, pending_state: Optional["ArrayDecodeRunner.PendingState"]
    ) -> None:
        if pending_state is None or pending_state.pending_steps == 0:
            if pending_state is not None:
                pending_state.reset()
            return
        if self._write_layers_tokens(pending_state):
            if pending_state.view.seq_ids:
                self.manager.bump_decode_lengths(
                    pending_state.view.seq_ids, pending_state.pending_steps
                )
            pending_state.reset()
            return
        steps = pending_state.pending_steps
        for step in range(steps):
            batches: list[Optional[ArrayDecodeRunner.WriterBatch]] = []
            for layer_idx in range(len(self.graph.layers)):
                layer_entries = pending_state.layer_batches[layer_idx]
                if step < len(layer_entries):
                    batches.append(layer_entries[step])
            self._flush_writer_batches(batches)
        if pending_state.view.seq_ids:
            self.manager.bump_decode_lengths(
                pending_state.view.seq_ids, pending_state.pending_steps
            )
        pending_state.reset()

    # ------------------------------------------------------------------#
    # Layer helpers
    # ------------------------------------------------------------------#
    def _run_layer(
        self,
        layer_idx: int,
        layer,
        hidden: mx.array,
        view: ArrayBatchView,
        *,
        pending_state: Optional["ArrayDecodeRunner.PendingState"] = None,
    ) -> tuple[mx.array, Optional["ArrayDecodeRunner.WriterBatch"]]:
        attn = layer.self_attn
        offsets = view.context_lens
        q, k, v = self._phase1_layer(layer_idx, layer, hidden, offsets)
        if _ARRAY_DEBUG and not self._logged_q_shape:
            _LOG.info(
                "ArrayDecodeRunner: q shape=%s k shape=%s view_rows=%s",
                q.shape,
                k.shape,
                view.active_rows,
            )
            self._logged_q_shape = True
        k_token = k[:, :, 0, :]
        v_token = v[:, :, 0, :]
        writer_batch = self._prepare_writer_batch(layer_idx, view, k_token, v_token)
        prefill_overlay = (
            view.overlay_for_layer(layer_idx)
            if hasattr(view, "overlay_for_layer")
            else None
        )
        overlay_k = mx.expand_dims(k_token, axis=0)
        overlay_v = mx.expand_dims(v_token, axis=0)
        overlay_len_value = 1
        overlay_len_arr = mx.array([overlay_len_value], dtype=mx.int32)
        if pending_state is not None:
            token_k = k_token[: view.active_rows]
            token_v = v_token[: view.active_rows]
            buf_k, buf_v = self._ensure_overlay_buffers(
                pending_state, layer_idx, token_k, token_v
            )
            write_idx = min(
                pending_state.overlay_lengths[layer_idx], pending_state.capacity - 1
            )
            buf_k = buf_k.at[write_idx].multiply(0)
            buf_k = buf_k.at[write_idx].add(token_k)
            buf_v = buf_v.at[write_idx].multiply(0)
            buf_v = buf_v.at[write_idx].add(token_v)
            pending_state.overlay_k_buffers[layer_idx] = buf_k
            pending_state.overlay_v_buffers[layer_idx] = buf_v
            pending_state.overlay_lengths[layer_idx] = min(
                write_idx + 1, pending_state.capacity
            )
            overlay_k = buf_k
            overlay_v = buf_v
            overlay_len_value = max(1, pending_state.overlay_lengths[layer_idx])
            overlay_len_arr = mx.array([overlay_len_value], dtype=mx.int32)
        overlay_rows: Tuple[int, ...] = ()
        if isinstance(prefill_overlay, PrefillOverlayView):
            overlay_rows = tuple(prefill_overlay.rows)
        if (
            prefill_overlay is not None
            and overlay_rows
            and len(overlay_rows) != view.active_rows
        ):
            hidden, wrote_in_graph = self._run_layer_partial(
                layer_idx=layer_idx,
                layer=layer,
                hidden=hidden,
                attn=attn,
                view=view,
                q=q,
                overlay_k=overlay_k,
                overlay_v=overlay_v,
                overlay_len_value=overlay_len_value,
                prefill_overlay=prefill_overlay,
                overlay_rows=overlay_rows,
            )
        else:
            combined_overlay_k = overlay_k
            combined_overlay_v = overlay_v
            combined_overlay_len = overlay_len_value
            overlay_context = None
            if isinstance(prefill_overlay, PrefillOverlayView):
                if combined_overlay_k is None:
                    combined_overlay_k = prefill_overlay.k
                    combined_overlay_v = prefill_overlay.v
                else:
                    combined_overlay_k = mx.concatenate(
                        [prefill_overlay.k, combined_overlay_k], axis=0
                    )
                    combined_overlay_v = mx.concatenate(
                        [prefill_overlay.v, combined_overlay_v], axis=0
                    )
                combined_overlay_len = int(combined_overlay_k.shape[0])
                overlay_context = prefill_overlay.base_lens
            overlay_len_arr = None
            if combined_overlay_k is not None:
                overlay_len_arr = mx.array(
                    [int(combined_overlay_k.shape[0])], dtype=mx.int32
                )
            hidden, wrote_in_graph = self._phase2_layer(
                layer_idx,
                layer,
                hidden,
                attn,
                view,
                q,
                combined_overlay_k,
                combined_overlay_v,
                overlay_len=combined_overlay_len,
                overlay_len_arr=overlay_len_arr,
                overlay_context=overlay_context,
            )
        if pending_state is not None and writer_batch is not None:
            pending_state.record_writer_batch(layer_idx, writer_batch)
            return hidden, None
        return hidden, None if wrote_in_graph else writer_batch

    def _subset_view(self, view: ArrayBatchView, rows: Sequence[int]) -> ArrayBatchView:
        if not rows:
            raise ValueError("subset view expects at least one row")
        row_indices = tuple(int(idx) for idx in rows)
        row_idx = mx.array(row_indices, dtype=mx.int32)
        block_tables = mx.take(view.block_tables, row_idx, axis=0)
        context_lens = mx.take(view.context_lens, row_idx, axis=0)
        write_blocks = mx.take(view.write_block_ids, row_idx, axis=0)
        write_offsets = mx.take(view.write_token_offsets, row_idx, axis=0)
        base_lens = (
            None
            if view.prefill_base_lens is None
            else mx.take(view.prefill_base_lens, row_idx, axis=0)
        )
        arrays = PagedDecodeArrays(
            seq_ids=tuple(view.seq_ids[idx] for idx in row_indices),
            block_tables=block_tables,
            context_lens=context_lens,
            kv_head_mapping=view.kv_head_mapping,
            k_cache=view._k_cache,
            v_cache=view._v_cache,
            quant_kwargs=list(view._quant_kwargs),
            write_block_ids=write_blocks,
            write_token_offsets=write_offsets,
            prefill_base_lens=base_lens,
            signature=view.signature,
            active_rows=len(row_indices),
        )
        return ArrayBatchView(arrays)

    def _run_layer_partial(
        self,
        *,
        layer_idx: int,
        layer,
        hidden: mx.array,
        attn,
        view: ArrayBatchView,
        q: mx.array,
        overlay_k: mx.array,
        overlay_v: mx.array,
        overlay_len_value: int,
        prefill_overlay: PrefillOverlayView,
        overlay_rows: Tuple[int, ...],
    ) -> tuple[mx.array, bool]:
        rows = tuple(int(idx) for idx in overlay_rows)
        rows_set = set(rows)
        wrote_any = False
        segments: list[Optional[mx.array]] = [None] * view.active_rows

        if rows:
            overlay_view = self._subset_view(view, rows)
            row_idx = mx.array(rows, dtype=mx.int32)
            overlay_hidden = mx.take(hidden, row_idx, axis=0)
            overlay_q = mx.take(q, row_idx, axis=0)
            if _ARRAY_DEBUG:
                _LOG.info(
                    "ArrayDecodeRunner: overlay rows=%s overlay_q_shape=%s overlay_k_shape=%s",
                    rows,
                    overlay_q.shape,
                    overlay_k.shape,
                )
            overlay_decode_k = mx.take(overlay_k, row_idx, axis=1)
            overlay_decode_v = mx.take(overlay_v, row_idx, axis=1)
            combined_k = overlay_decode_k
            combined_v = overlay_decode_v
            overlay_context = None
            combined_len = overlay_len_value
            if (
                isinstance(prefill_overlay, PrefillOverlayView)
                and prefill_overlay.tokens > 0
            ):
                combined_k = mx.concatenate(
                    [prefill_overlay.k, overlay_decode_k], axis=0
                )
                combined_v = mx.concatenate(
                    [prefill_overlay.v, overlay_decode_v], axis=0
                )
                combined_len = int(combined_k.shape[0])
                overlay_context = prefill_overlay.base_lens
            overlay_len_arr = mx.array([combined_len], dtype=mx.int32)
            overlay_hidden, wrote_overlay = self._phase2_layer(
                layer_idx,
                layer,
                overlay_hidden,
                attn,
                overlay_view,
                overlay_q,
                combined_k,
                combined_v,
                overlay_len=combined_len,
                overlay_len_arr=overlay_len_arr,
                overlay_context=overlay_context,
            )
            for pos, row in enumerate(rows):
                segments[row] = overlay_hidden[pos : pos + 1]
            wrote_any = wrote_any or wrote_overlay
        plain_rows = tuple(
            idx for idx in range(view.active_rows) if idx not in rows_set
        )
        if plain_rows:
            plain_view = self._subset_view(view, plain_rows)
            plain_idx = mx.array(plain_rows, dtype=mx.int32)
            plain_hidden = mx.take(hidden, plain_idx, axis=0)
            plain_q = mx.take(q, plain_idx, axis=0)
            plain_decode_k = mx.take(overlay_k, plain_idx, axis=1)
            plain_decode_v = mx.take(overlay_v, plain_idx, axis=1)
            plain_len_arr = mx.array([overlay_len_value], dtype=mx.int32)
            plain_hidden, wrote_plain = self._phase2_layer(
                layer_idx,
                layer,
                plain_hidden,
                attn,
                plain_view,
                plain_q,
                plain_decode_k,
                plain_decode_v,
                overlay_len=overlay_len_value,
                overlay_len_arr=plain_len_arr,
                overlay_context=None,
            )
            for pos, row in enumerate(plain_rows):
                segments[row] = plain_hidden[pos : pos + 1]
            wrote_any = wrote_any or wrote_plain
        ordered = [
            segment if segment is not None else hidden[row : row + 1]
            for row, segment in enumerate(segments)
        ]
        merged = mx.concatenate(ordered, axis=0)
        return merged, wrote_any

    def _apply_attention(
        self,
        layer_idx: int,
        attn,
        queries: mx.array,
        view: ArrayBatchView,
        *,
        k_overlay: Optional[mx.array] = None,
        v_overlay: Optional[mx.array] = None,
        overlay_len: Optional[mx.array] = None,
        context_override: Optional[mx.array] = None,
    ) -> mx.array:
        (
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            _,
            kv_head_mapping,
            quant_kwargs,
        ) = view.args_for_layer(layer_idx)
        context_values = (
            context_override if context_override is not None else context_lens
        )
        overlay_len_arr = None
        if overlay_len is not None:
            if isinstance(overlay_len, mx.array):
                overlay_len_arr = overlay_len
            else:
                overlay_len_arr = mx.array([int(overlay_len)], dtype=mx.int32)
        elif k_overlay is not None:
            overlay_len_arr = mx.array([int(k_overlay.shape[0])], dtype=mx.int32)
        attn_kwargs = dict(quant_kwargs or {})
        if kv_head_mapping is not None:
            attn_kwargs["kv_head_mapping"] = kv_head_mapping
        attn_kwargs["scale"] = attn.scale
        attn_kwargs.setdefault("layer_idx", layer_idx)
        context_plus_one = context_lens + 1
        if k_overlay is not None and v_overlay is not None:
            layer_kw = attn_kwargs.get("layer_idx", layer_idx)
            overlay_kwargs = dict(attn_kwargs)
            overlay_kwargs.pop("layer_idx", None)
            return mx.fast._paged_attention_with_overlay_impl(
                queries,
                k_cache,
                v_cache,
                block_tables,
                context_values,
                layer_idx=layer_kw,
                k_overlay=k_overlay,
                v_overlay=v_overlay,
                overlay_len=overlay_len_arr,
                **overlay_kwargs,
            )
        return mx.fast.paged_attention(
            queries,
            k_cache,
            v_cache,
            block_tables,
            context_lens + 1,
            **attn_kwargs,
        )

    def _prepare_writer_batch(
        self,
        layer_idx: int,
        view: ArrayBatchView,
        k_token: mx.array,
        v_token: mx.array,
    ) -> Optional["ArrayDecodeRunner.WriterBatch"]:
        active = view.active_rows
        if active <= 0:
            return None
        (
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            _,
            _kv_head_mapping,
            quant_kwargs,
        ) = view.args_for_layer(layer_idx)

        block_tables_slice = block_tables
        context_slice = mx.array(context_lens)
        k_batch = mx.array(k_token[:active])
        v_batch = mx.array(v_token[:active])
        return ArrayDecodeRunner.WriterBatch(
            layer_idx=layer_idx,
            k_cache=k_cache,
            v_cache=v_cache,
            seq_ids=view.seq_ids,
            block_tables=block_tables_slice,
            context_lens=context_slice,
            k_batch=k_batch,
            v_batch=v_batch,
            quant_kwargs=quant_kwargs,
            active_rows=active,
        )

    def _write_tokens(
        self,
        writer_batch: Optional["ArrayDecodeRunner.WriterBatch"],
    ) -> None:
        if writer_batch is None:
            return
        if writer_batch.active_rows <= 0:
            return
        start = time.perf_counter()
        quant_enabled = bool(writer_batch.quant_kwargs)

        try:
            if quant_enabled:
                self._write_tokens_fallback(
                    writer_batch.layer_idx,
                    writer_batch.seq_ids[: writer_batch.active_rows],
                    writer_batch.context_lens,
                    writer_batch.k_batch,
                    writer_batch.v_batch,
                )
                return

            mx.fast.paged_kv_write_batch(
                writer_batch.k_cache,
                writer_batch.v_cache,
                writer_batch.block_tables,
                writer_batch.context_lens,
                writer_batch.k_batch,
                writer_batch.v_batch,
            )
        except RuntimeError:
            _LOG.warning(
                "ArrayDecodeRunner: batch write fallback layer=%s active=%s",
                writer_batch.layer_idx,
                writer_batch.active_rows,
                exc_info=True,
            )
            self._write_tokens_fallback(
                writer_batch.layer_idx,
                writer_batch.seq_ids[: writer_batch.active_rows],
                writer_batch.context_lens,
                writer_batch.k_batch,
                writer_batch.v_batch,
            )
        finally:
            self._writer_duration_s += time.perf_counter() - start

    def _write_tokens_fallback(
        self,
        layer_idx: int,
        seq_ids: Sequence[int],
        context_lens: mx.array,
        k_batch: mx.array,
        v_batch: mx.array,
    ) -> None:
        for row, seq_id in enumerate(seq_ids[: context_lens.shape[0]]):
            k_chunk = mx.expand_dims(k_batch[row], axis=1)
            v_chunk = mx.expand_dims(v_batch[row], axis=1)
            start_pos = int(context_lens[row].item())
            self.manager._write_tokens(
                seq_id, layer_idx, k_chunk, v_chunk, start_pos, commit=False
            )

    def _flush_writer_batches(
        self, batches: Sequence[Optional["ArrayDecodeRunner.WriterBatch"]]
    ) -> None:
        pending = [b for b in batches if b is not None]
        if not pending:
            return
        if self._write_layers_batch(pending):
            return
        for batch in pending:
            self._write_tokens(batch)

    def _write_layers_batch(
        self, batches: Sequence["ArrayDecodeRunner.WriterBatch"]
    ) -> bool:
        if not batches:
            return False
        if any(batch.quant_kwargs for batch in batches):
            return False
        active = batches[0].active_rows
        seq_ids = tuple(batches[0].seq_ids[:active])
        ref_tables = batches[0].block_tables
        ref_context = batches[0].context_lens
        for batch in batches[1:]:
            if (
                batch.active_rows != active
                or batch.block_tables.shape != ref_tables.shape
                or batch.context_lens.shape != ref_context.shape
            ):
                return False
        k_layers = mx.stack([batch.k_batch for batch in batches], axis=0)
        v_layers = mx.stack([batch.v_batch for batch in batches], axis=0)
        start = time.perf_counter()
        try:
            mx.fast.paged_kv_write_layers_batch(
                self.manager.k,
                self.manager.v,
                ref_tables,
                ref_context,
                k_layers,
                v_layers,
            )
            self._writer_duration_s += time.perf_counter() - start
            return True
        except Exception:
            _LOG.warning(
                "ArrayDecodeRunner: layered writer failed; falling back to per-layer path",
                exc_info=True,
            )
            return False

    def _write_layers_tokens(
        self, pending_state: "ArrayDecodeRunner.PendingState"
    ) -> bool:
        steps = pending_state.pending_steps
        if steps <= 0:
            return True
        layer_batches = pending_state.layer_batches
        if not layer_batches or not layer_batches[0]:
            return False
        reference_batch = None
        for batches in layer_batches:
            if batches:
                reference_batch = batches[0]
                break
        if reference_batch is None:
            return False
        if reference_batch.quant_kwargs:
            return False
        active = reference_batch.active_rows
        if active <= 0:
            return True
        seq_ids = tuple(reference_batch.seq_ids[:active])
        block_tables = reference_batch.block_tables
        context_lens = reference_batch.context_lens
        k_layers: list[mx.array] = []
        v_layers: list[mx.array] = []
        for layer_idx, batches in enumerate(layer_batches):
            if len(batches) != steps:
                return False
            if any(batch is None or batch.quant_kwargs for batch in batches):
                return False
            tokens_k = []
            tokens_v = []
            for batch in batches:
                if batch.active_rows != active:
                    return False
                if tuple(batch.seq_ids[:active]) != seq_ids:
                    return False
                tokens_k.append(mx.expand_dims(batch.k_batch[:active], axis=0))
                tokens_v.append(mx.expand_dims(batch.v_batch[:active], axis=0))
            k_layers.append(mx.concatenate(tokens_k, axis=0))
            v_layers.append(mx.concatenate(tokens_v, axis=0))
        k_tensor = mx.stack(k_layers, axis=0)
        v_tensor = mx.stack(v_layers, axis=0)
        start = time.perf_counter()
        try:
            mx.fast.paged_kv_write_layers_tokens(
                self.manager.k,
                self.manager.v,
                block_tables[:active],
                context_lens[:active],
                k_tensor,
                v_tensor,
            )
            self._writer_duration_s += time.perf_counter() - start
            return True
        except Exception:
            self._writer_duration_s += time.perf_counter() - start
            _LOG.warning(
                "ArrayDecodeRunner: multi-token layered writer failed; falling back",
                exc_info=True,
            )
            return False

    def _phase1_layer(
        self, layer_idx: int, layer, hidden: mx.array, offsets: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        cache = self._phase1_layers.setdefault(layer_idx, {})
        shape_key = (
            tuple(hidden.shape),
            tuple(offsets.shape),
            hidden.dtype,
            offsets.dtype,
        )
        fn = cache.get(shape_key)
        if fn is None:
            fn = self._build_phase1_layer_fn(layer_idx, layer, hidden, offsets)
            cache[shape_key] = fn
        start = time.perf_counter()
        try:
            return fn(hidden, offsets)
        finally:
            self._phase1_duration_s += time.perf_counter() - start

    def _build_phase1_layer_fn(
        self,
        layer_idx: int,
        layer,
        hidden: mx.array,
        offsets: mx.array,
    ):
        attn = layer.self_attn

        def eager(hidden: mx.array, offsets: mx.array):
            normed = layer.input_layernorm(hidden)
            qkv = self.graph.project_qkv(attn, normed)
            q = self.graph.apply_rope(attn, qkv.q, offsets)
            k = self.graph.apply_rope(attn, qkv.k, offsets)
            v = self.graph.apply_rope(attn, qkv.v, offsets)
            return q, k, v

        if self.decode_engine == "paged-arrays+compile" and hasattr(mx, "compile"):
            _LOG.info("ArrayDecodeRunner: attempting compile for layer %s", layer_idx)
            try:
                spec_hidden = mx.zeros(hidden.shape, dtype=hidden.dtype)
                spec_offsets = mx.zeros(offsets.shape, dtype=offsets.dtype)
                compiled = mx.compile(eager, inputs=[spec_hidden, spec_offsets])
                _LOG.info("ArrayDecodeRunner: compiled phase-1 for layer %s", layer_idx)
                self._phase1_compile_misses += 1

                def compiled_wrapper(hidden, offsets):
                    self._phase1_compile_hits += 1
                    return compiled(hidden, offsets)

                return compiled_wrapper
            except Exception:
                _LOG.warning(
                    "ArrayDecodeRunner: unable to compile phase-1 for layer %s",
                    layer_idx,
                    exc_info=True,
                )
        return eager

    def _get_phase2_compiled_fn(
        self,
        layer_idx: int,
        layer,
        attn,
        hidden: mx.array,
        q: mx.array,
        k_overlay: mx.array,
        v_overlay: mx.array,
        block_tables: mx.array,
        context_lens: mx.array,
        k_cache: mx.array,
        v_cache: mx.array,
        kv_head_mapping: Optional[mx.array],
        quant_kwargs: dict,
    ) -> tuple[callable, tuple]:
        quant_enabled = bool(quant_kwargs)
        kv_map_enabled = kv_head_mapping is not None
        quant_sig = None
        if quant_enabled:
            quant_sig = (
                tuple(quant_kwargs["v_q_cache"].shape),
                str(quant_kwargs["v_q_cache"].dtype),
                tuple(quant_kwargs["v_scale_cache"].shape),
                str(quant_kwargs["v_scale_cache"].dtype),
                tuple(quant_kwargs["v_zero_cache"].shape),
                str(quant_kwargs["v_zero_cache"].dtype),
                int(quant_kwargs["quant_bits"]),
                int(quant_kwargs["quant_group_size"]),
                int(quant_kwargs["quant_bytes_per_token"]),
                int(quant_kwargs["quant_groups_per_head"]),
                bool(quant_kwargs.get("quant_symmetric", False)),
            )
        key = (
            tuple(hidden.shape),
            str(hidden.dtype),
            tuple(q.shape),
            str(q.dtype),
            tuple(k_overlay.shape),
            tuple(v_overlay.shape),
            tuple(block_tables.shape),
            tuple(context_lens.shape),
            tuple(k_cache.shape),
            str(k_cache.dtype),
            str(v_cache.dtype),
            kv_map_enabled,
            quant_sig,
        )
        cache = self._phase2_layers.setdefault(layer_idx, {})
        entry = cache.get(key)
        if entry is None:
            compiled = self._build_phase2_compiled_fn(
                layer_idx=layer_idx,
                layer=layer,
                attn=attn,
                hidden=hidden,
                q=q,
                k_overlay=k_overlay,
                v_overlay=v_overlay,
                block_tables=block_tables,
                context_lens=context_lens,
                k_cache=k_cache,
                v_cache=v_cache,
                kv_head_mapping=kv_head_mapping,
                quant_kwargs=quant_kwargs if quant_enabled else None,
            )
            entry = (compiled, kv_map_enabled, quant_enabled)
            cache[key] = entry
            self._phase2_compile_misses += 1
        else:
            self._phase2_compile_hits += 1
        compiled_fn, kv_flag, quant_flag = entry
        extras = []
        if kv_flag:
            extras.append(kv_head_mapping)
        if quant_flag:
            extras.extend(
                [
                    quant_kwargs["v_q_cache"],
                    quant_kwargs["v_scale_cache"],
                    quant_kwargs["v_zero_cache"],
                ]
            )
        return compiled_fn, tuple(extras)

    def _build_phase2_compiled_fn(
        self,
        layer_idx: int,
        layer,
        attn,
        hidden: mx.array,
        q: mx.array,
        k_overlay: mx.array,
        v_overlay: mx.array,
        block_tables: mx.array,
        context_lens: mx.array,
        k_cache: mx.array,
        v_cache: mx.array,
        kv_head_mapping: Optional[mx.array],
        quant_kwargs: Optional[dict],
    ):
        specs = [
            mx.zeros(hidden.shape, dtype=hidden.dtype),
            mx.zeros(q.shape, dtype=q.dtype),
            mx.zeros(k_overlay.shape, dtype=k_overlay.dtype),
            mx.zeros(v_overlay.shape, dtype=v_overlay.dtype),
            mx.zeros(block_tables.shape, dtype=block_tables.dtype),
            mx.zeros(context_lens.shape, dtype=context_lens.dtype),
            mx.zeros(k_cache.shape, dtype=k_cache.dtype),
            mx.zeros(v_cache.shape, dtype=v_cache.dtype),
            mx.zeros((1,), dtype=mx.int32),
        ]
        quant_meta = None
        kv_map_enabled = kv_head_mapping is not None
        if kv_map_enabled:
            specs.append(mx.zeros(kv_head_mapping.shape, dtype=kv_head_mapping.dtype))
        if quant_kwargs is not None:
            specs.extend(
                [
                    mx.zeros(
                        quant_kwargs["v_q_cache"].shape,
                        dtype=quant_kwargs["v_q_cache"].dtype,
                    ),
                    mx.zeros(
                        quant_kwargs["v_scale_cache"].shape,
                        dtype=quant_kwargs["v_scale_cache"].dtype,
                    ),
                    mx.zeros(
                        quant_kwargs["v_zero_cache"].shape,
                        dtype=quant_kwargs["v_zero_cache"].dtype,
                    ),
                ]
            )
            quant_meta = {
                "quant_bits": int(quant_kwargs["quant_bits"]),
                "quant_group_size": int(quant_kwargs["quant_group_size"]),
                "quant_bytes_per_token": int(quant_kwargs["quant_bytes_per_token"]),
                "quant_groups_per_head": int(quant_kwargs["quant_groups_per_head"]),
                "quant_symmetric": bool(quant_kwargs.get("quant_symmetric", False)),
            }

        attn_kwargs = {"layer_idx": layer_idx, "scale": attn.scale}

        def phase2_impl(
            hidden,
            q,
            k_overlay,
            v_overlay,
            block_tables,
            context_lens,
            k_cache,
            v_cache,
            overlay_len_arr,
            *extras,
        ):
            idx = 0
            local_kwargs = dict(attn_kwargs)
            kv_map = None
            if kv_map_enabled:
                kv_map = extras[idx]
                idx += 1
                local_kwargs["kv_head_mapping"] = kv_map
            if quant_meta is not None:
                v_q = extras[idx]
                v_scale = extras[idx + 1]
                v_zero = extras[idx + 2]
                idx += 3
                local_kwargs.update(
                    v_q_cache=v_q,
                    v_scale_cache=v_scale,
                    v_zero_cache=v_zero,
                    quant_bits=quant_meta["quant_bits"],
                    quant_group_size=quant_meta["quant_group_size"],
                    quant_bytes_per_token=quant_meta["quant_bytes_per_token"],
                    quant_groups_per_head=quant_meta["quant_groups_per_head"],
                    quant_symmetric=quant_meta["quant_symmetric"],
                )
            attn_ctx = mx.fast._paged_attention_with_overlay_impl(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                k_overlay=k_overlay,
                v_overlay=v_overlay,
                overlay_len=overlay_len_arr,
                **local_kwargs,
            )
            attn_out = self.graph.attention_output(attn, attn_ctx)
            hidden_out = hidden + attn_out
            hidden_out = hidden_out + self.graph.mlp(layer, hidden_out)
            return hidden_out

        compiled = mx.compile(phase2_impl, inputs=specs)
        return compiled

    def _phase2_layer(
        self,
        layer_idx: int,
        layer,
        hidden: mx.array,
        attn,
        view: ArrayBatchView,
        q: mx.array,
        k_overlay: mx.array,
        v_overlay: mx.array,
        *,
        overlay_len: Optional[int] = None,
        overlay_len_arr: Optional[mx.array] = None,
        overlay_context: Optional[mx.array] = None,
    ) -> tuple[mx.array, bool]:
        (
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            _,
            kv_head_mapping,
            quant_kwargs,
        ) = view.args_for_layer(layer_idx)
        context_source = (
            overlay_context if overlay_context is not None else context_lens
        )

        use_compile = self.decode_engine == "paged-arrays+compile" and hasattr(
            mx, "compile"
        )

        if use_compile:
            compiled_fn, extra = self._get_phase2_compiled_fn(
                layer_idx,
                layer,
                attn,
                hidden,
                q,
                k_overlay,
                v_overlay,
                block_tables,
                context_source,
                k_cache,
                v_cache,
                kv_head_mapping,
                quant_kwargs,
            )
            start = time.perf_counter()
            hidden = compiled_fn(
                hidden,
                q,
                k_overlay,
                v_overlay,
                block_tables,
                context_source,
                k_cache,
                v_cache,
                overlay_len_arr,
                *extra,
            )
            self._phase2_compiled_duration_s += time.perf_counter() - start
            return hidden, False

        start = time.perf_counter()
        attn_ctx = self._apply_attention(
            layer_idx,
            attn,
            q,
            view,
            k_overlay=k_overlay,
            v_overlay=v_overlay,
            overlay_len=overlay_len_arr,
            context_override=context_source,
        )
        self._attn_duration_s += time.perf_counter() - start

        start = time.perf_counter()
        attn_out = self.graph.attention_output(attn, attn_ctx)
        hidden = hidden + attn_out
        self._outproj_duration_s += time.perf_counter() - start

        start = time.perf_counter()
        hidden = hidden + self.graph.mlp(layer, hidden)
        self._mlp_duration_s += time.perf_counter() - start

        return hidden, False

    def compile_stats(self) -> dict[str, float]:
        return {
            "array_phase1_compile_hits": float(self._phase1_compile_hits),
            "array_phase1_compile_misses": float(self._phase1_compile_misses),
            "array_phase2_compile_hits": float(self._phase2_compile_hits),
            "array_phase2_compile_misses": float(self._phase2_compile_misses),
            "array_phase1_duration_s": float(self._phase1_duration_s),
            "array_phase2_attention_duration_s": float(self._attn_duration_s),
            "array_phase2_outproj_duration_s": float(self._outproj_duration_s),
            "array_phase2_mlp_duration_s": float(self._mlp_duration_s),
            "array_writer_duration_s": float(self._writer_duration_s),
            "array_phase2_compiled_duration_s": float(self._phase2_compiled_duration_s),
        }


__all__ = ["ArrayDecodeRunner"]
