# ABOUTME: Implements continuous batching engine bridging slot generator modes.
# ABOUTME: Provides ModelRunner wrapper that maps model outputs to contexts.

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

try:  # pragma: no cover - optional dependency when MLX lacks paged_kv
    from mlx.nn.paged_kv import KVBlockManager, QuantSpec
except Exception:  # pragma: no cover
    KVBlockManager = None
    QuantSpec = None

from ..generate import BatchGenerator, GenerationResponse
from ..sample_utils import make_logits_processors, make_sampler
from .paged_attention_patch import install_paged_attention_patch
from .paged_context import wrap_attention_layers
from .paged_slot_kv_cache import BackendSignature, PagedSlotKVCache, PrefixCache
from .safe_eval import safe_eval
from .slot_allocator import SlotAllocator
from .slot_batcher import SlotBatcher
from .slot_generator import SlotGenerator
from .slot_kv_cache import SlotKVCache
from .state import SequenceContext, SequenceState

_MODEL_CHILD_ATTRS = ("model", "llm", "transformer", "decoder", "module")
_DEFAULT_KV_POOL_BLOCKS = 8192
_AUTO_KV_POOL_FRACTION = 0.7
_AUTO_KV_POOL_MAX_BLOCKS = max(
    1,
    int(os.environ.get("MLXLM_AUTO_KV_MAX_BLOCKS", str(_DEFAULT_KV_POOL_BLOCKS))),
)
_MEMORY_WARN_FRACTION = 0.8
_MEMORY_RESET_FRACTION = 0.7
_MX_ARRAY_TYPE = type(mx.array(0))


class ModelRunner:
    """Encapsulates model execution for the continuous batching scheduler."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        *,
        draft_model: Optional[nn.Module] = None,
        max_num_seqs: int = 16,
        prefill_chunk: int = 1024,
        prefill_ramp_chunk: Optional[int] = None,
        prefill_hybrid_threshold: int = 0,
        force_legacy_generator: bool = False,
        stop_tokens: Optional[Iterable[int]] = None,
        attn_backend: str = "dense",
        kv_block_size: int = 16,
        kv_pool_blocks: Optional[int] = None,
        paged_vec_width: Optional[int] = None,
        paged_threads_per_head: Optional[int] = None,
        kv_quant_mode: str = "none",
        kv_quant_group_size: int = 64,
        decode_unroll: int = 1,
        decode_unroll_safe: bool = True,
        decode_engine: str = "dense",
        prefill_ramp_budget_ms: Optional[float] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.max_num_seqs = max_num_seqs
        self.prefill_chunk = prefill_chunk
        self.prefill_ramp_chunk = self._normalize_prefill_ramp(prefill_ramp_chunk)
        if prefill_ramp_budget_ms is not None and prefill_ramp_budget_ms <= 0:
            prefill_ramp_budget_ms = None
        self.prefill_ramp_budget_ms: Optional[float] = (
            float(prefill_ramp_budget_ms) if prefill_ramp_budget_ms else None
        )
        if prefill_hybrid_threshold is None:
            prefill_hybrid_threshold = 0
        self.prefill_hybrid_threshold = max(
            0, min(int(prefill_hybrid_threshold), self.prefill_chunk)
        )
        self.use_legacy_generator = force_legacy_generator
        self.requested_attn_backend = (attn_backend or "dense").lower()
        self.actual_attn_backend = "dense"
        self.kv_block_size = max(1, int(kv_block_size or 1))
        self.requested_kv_pool_blocks = kv_pool_blocks
        self.kv_pool_blocks = (
            None if kv_pool_blocks is None else max(1, int(kv_pool_blocks))
        )
        self.kv_pool_meta: Dict[str, float] = {}
        self.paged_vec_width = paged_vec_width
        self.paged_threads_per_head = paged_threads_per_head
        self.kv_quant_mode = (kv_quant_mode or "none").lower()
        self.kv_quant_group_size = max(1, int(kv_quant_group_size or 1))
        self.decode_unroll = max(1, int(decode_unroll or 1))
        self.decode_unroll_safe = bool(decode_unroll_safe)
        self.decode_engine = decode_engine
        self._forced_dense_for_quant = False
        if (
            self.kv_quant_mode != "none"
            and isinstance(self.decode_engine, str)
            and "paged" in self.decode_engine
            and os.environ.get("MLXLM_ALLOW_PAGED_KV_QUANT", "").lower()
            not in {"1", "true", "yes"}
        ):
            logging.warning(
                "Paged decode does not support kv_quant_mode=%s on this build; forcing dense attention with unquantized KV",
                self.kv_quant_mode,
            )
            self.kv_quant_mode = "none"
            self.decode_engine = "dense"
            self._forced_dense_for_quant = True
            self.requested_attn_backend = "dense"
        logging.info(
            "ModelRunner: requested_attn_backend=%s decode_engine=%s",
            self.requested_attn_backend,
            self.decode_engine,
        )
        self.kv_manager: Optional[KVBlockManager] = None
        self.kv_geometry: Optional[dict] = None
        self.kv_head_mapping: Optional[mx.array] = None
        self.kv_dtype = None
        self.paged_backend_enabled = False
        self.paged_cache: Optional[PagedSlotKVCache] = None
        self._prefill_calls = 0
        self._prefill_tokens = 0
        self._prefill_duration_s = 0.0
        self._last_decode_stats = {
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
            "prefill_calls": 0,
            "prefill_tokens": 0,
            "prefill_duration_s": 0.0,
        }
        self._last_active_count = 0
        self._memory_warning_emitted = False

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if stop_tokens is not None:
            stop_tokens_set = {int(tok) for tok in stop_tokens}
        else:
            stop_tokens_set = set()
            if eos_id is not None:
                if isinstance(eos_id, (list, tuple, set)):
                    stop_tokens_set.update(int(tok) for tok in eos_id)
                else:
                    stop_tokens_set.add(int(eos_id))
            eos_ids = getattr(tokenizer, "eos_token_ids", None)
            if eos_ids:
                stop_tokens_set.update(int(tok) for tok in eos_ids)

        self.stop_tokens = set(stop_tokens_set)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        xtc_special = (
            list(self.stop_tokens)
            if self.stop_tokens
            else ([eos_id] if eos_id is not None else [])
        )

        self._default_sampler_config = {
            "temp": 0.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": 0,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.0,
            "xtc_special_tokens": xtc_special,
        }
        self.default_sampler = make_sampler(**self._default_sampler_config)

        overlap_env = os.environ.get("MLXLM_PREFILL_OVERLAP")
        if overlap_env is None:
            self.prefill_buffer = max_num_seqs
        else:
            try:
                self.prefill_buffer = max(0, int(overlap_env))
            except ValueError:
                self.prefill_buffer = max_num_seqs
        self.total_slots = max_num_seqs + self.prefill_buffer
        slice_env = os.environ.get("MLXLM_PREFILL_SLICE_TOKENS")
        slice_target = 64
        if slice_env:
            try:
                slice_target = max(1, int(slice_env))
            except ValueError:
                slice_target = 64
        self.prefill_slice_cap = max(1, min(prefill_chunk, slice_target))
        # Slot scaffolding for upcoming vectorized decode.
        self.slot_allocator = SlotAllocator(self.total_slots)
        # Placeholder KV cache capacity equal to prefill chunk for stage scaffolding.
        self.slot_kv_cache = SlotKVCache(
            max_slots=self.total_slots,
            capacity_tokens=prefill_chunk or 1,
            kv_heads=1,
            head_dim=1,
        )
        self.slot_batcher = SlotBatcher(self.slot_allocator, self.slot_kv_cache)
        self.uid_to_context: Dict[int, SequenceContext] = {}
        self.decode_buckets = self._compute_decode_buckets(max_num_seqs)
        self._decode_warm = False
        self.prefix_cache = None
        self._model_signature = getattr(model, "name", model.__class__.__name__)

        if self.use_legacy_generator:
            self.legacy_generator = BatchGenerator(
                model,
                stop_tokens=self.stop_tokens,
                completion_batch_size=max_num_seqs,
                prefill_batch_size=max_num_seqs,
                prefill_step_size=prefill_chunk,
                sampler=self._batched_sampler,
            )
            self.slot_generator = None
        else:
            self.legacy_generator = None
            self.slot_generator = SlotGenerator(
                model=model,
                tokenizer=tokenizer,
                slot_alloc=self.slot_allocator,
                prefill_chunk=prefill_chunk,
                decode_engine=decode_engine,
                prefill_ramp_chunk=self.prefill_ramp_chunk,
                prefill_hybrid_threshold=self.prefill_hybrid_threshold,
                prefill_ramp_budget_ms=self.prefill_ramp_budget_ms,
            )
            self.slot_generator.kv_block_size = self.kv_block_size
            self._prewarm_decode_buckets()

        if self.requested_attn_backend == "paged":
            logging.info("ModelRunner: initializing paged backend")
            self._initialize_paged_backend()

    def _normalize_prefill_ramp(self, value: Optional[int]) -> int:
        if value is None:
            return min(self.prefill_chunk, 64)
        ramp = int(value)
        if ramp <= 0:
            return 0
        return max(1, min(ramp, self.prefill_chunk))

    def _build_quant_spec(self):
        if QuantSpec is None:
            return None
        if self.kv_quant_mode != "int4_v":
            return None
        try:
            return QuantSpec(
                bits=4,
                group_size=self.kv_quant_group_size,
                targets=("v",),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------#
    # Context preparation
    # ------------------------------------------------------------------#
    def build_context(
        self,
        request_id: str,
        prompt: Sequence[int],
        *,
        max_new_tokens: int,
        sampler_settings: Dict[str, float],
        stopping_settings: Dict[str, Optional[int]],
        logit_bias: Optional[Dict[int, float]] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
    ) -> SequenceContext:
        prompt_tokens = list(prompt)
        sampler_cfg = dict(sampler_settings)
        stopping_cfg = dict(stopping_settings)
        state = SequenceState(
            request_id=request_id,
            prompt_len=len(prompt_tokens),
            max_new_tokens=max_new_tokens,
        )
        state.hybrid_dense_remaining = self.prefill_hybrid_threshold
        state.prefill_ramp_remaining = self.prefill_ramp_chunk
        if self.prefix_cache is not None:
            state.apc_prefixes = self._compute_apc_prefixes(prompt_tokens)
        sampler = make_sampler(**sampler_cfg)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=prompt_tokens,
            sampler_settings=sampler_cfg,
            stopping_settings=stopping_cfg,
            tokenizer=self.tokenizer,
            sampler=sampler,
            logits_processors=logits_processors or None,
        )
        ctx.uses_default_sampler = sampler_cfg == self._default_sampler_config
        if self.use_legacy_generator:
            ctx.detokenizer = self.tokenizer.detokenizer
        if prompt_tokens:
            ctx.last_token_id = prompt_tokens[-1]
        else:
            ctx.last_token_id = getattr(
                self.tokenizer, "bos_token_id", self.tokenizer.eos_token_id
            )
        ctx.history_tokens = list(prompt_tokens)
        ctx.stop_sequences = list(stopping_cfg.get("stop_id_sequences", []))
        ctx.stop_tokens = set(self.stop_tokens)
        return ctx

    def _compute_decode_buckets(self, max_num_seqs: int) -> List[int]:
        if max_num_seqs <= 0:
            return [1]
        fractions = (1.0, 0.75, 0.5, 0.25, 0.125)
        buckets = {max(1, int(round(max_num_seqs * frac))) for frac in fractions}
        buckets.add(1)
        return sorted(buckets, reverse=True)

    def _prewarm_decode_buckets(self) -> None:
        if self.use_legacy_generator or self._decode_warm:
            return
        try:
            call_sig = inspect.signature(self.model.__call__)
            if "cache" not in call_sig.parameters:
                self._decode_warm = True
                return
        except (TypeError, ValueError, AttributeError):
            self._decode_warm = True
            return
        try:
            for bucket in self.decode_buckets:
                if bucket <= 0:
                    continue
                contexts = [
                    self._make_padding_context(bucket, idx) for idx in range(bucket)
                ]
                try:
                    self.slot_generator.decode_step(contexts)
                finally:
                    for ctx in contexts:
                        ctx.state.finished = True
                        self.slot_generator.on_release(ctx)
        finally:
            self._decode_warm = True

    def _make_padding_context(self, bucket: int, index: int) -> SequenceContext:
        state = SequenceState(
            request_id=f"_warmup_{bucket}_{index}",
            prompt_len=1,
            max_new_tokens=1,
        )
        state.hybrid_dense_remaining = 0
        state.prefill_ramp_remaining = 0
        ctx = SequenceContext(
            state=state,
            prompt_tokens=[0],
            sampler_settings={},
            stopping_settings={},
            tokenizer=self.tokenizer,
        )
        ctx.history_tokens = [0]
        ctx.last_token_id = getattr(self.tokenizer, "bos_token_id", 0)
        admitted = self.slot_generator.on_admit(ctx)
        if not admitted:
            raise RuntimeError("Unable to admit warmup context; insufficient slots.")
        remaining = ctx.state.remaining_prompt_tokens
        if remaining > 0:
            self.slot_generator.prefill_tokens(ctx, remaining)
        return ctx

    # ------------------------------------------------------------------#
    # Prefill / decode
    # ------------------------------------------------------------------#
    def begin_step(self) -> None:
        self._prefill_calls = 0
        self._prefill_tokens = 0
        self._prefill_duration_s = 0.0
        self._last_decode_stats = {
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
            "prefill_calls": 0,
            "prefill_tokens": 0,
            "prefill_duration_s": 0.0,
        }
        self._last_active_count = 0

    def prefill_context(self, ctx: SequenceContext, num_tokens: int) -> int:
        if ctx.state.finished or num_tokens <= 0:
            return 0
        tick_start = time.perf_counter()
        if ctx.prefill_started_ns is None:
            ctx.prefill_started_ns = time.time_ns()

        processed = 0
        if self.use_legacy_generator:
            ctx.state.prompt_pos = min(
                ctx.state.prompt_pos + num_tokens, ctx.state.prompt_len
            )
            if (
                ctx.state.remaining_prompt_tokens == 0
                and not ctx.prompt_inserted
                and ctx.prompt_tokens
            ):
                uid = self.legacy_generator.insert(
                    [ctx.prompt_tokens], ctx.state.max_new_tokens
                )[0]
                ctx.generator_uid = uid
                ctx.prompt_inserted = True
                ctx.prefill_completed_ns = time.time_ns()
                self.uid_to_context[uid] = ctx
            processed = num_tokens
        else:
            if not ctx.slot_initialized:
                admitted = self.slot_generator.on_admit(ctx)
                if not admitted:
                    return 0
            processed = self.slot_generator.prefill_tokens(ctx, num_tokens)
            if (
                processed > 0
                and ctx.state.remaining_prompt_tokens == 0
                and ctx.prefill_completed_ns is None
            ):
                ctx.prefill_completed_ns = time.time_ns()

        tick_duration = time.perf_counter() - tick_start
        self._prefill_calls += 1
        self._prefill_tokens += max(processed, 0)
        self._prefill_duration_s += max(tick_duration, 0.0)
        return processed

    def prefill_context_batch(
        self, contexts: Sequence[SequenceContext], chunk_len: int
    ) -> List[int]:
        if chunk_len <= 0 or not contexts:
            return [0 for _ in contexts]
        tick_start = time.perf_counter()
        processed = [0 for _ in contexts]
        prepared: List[SequenceContext] = []
        for idx, ctx in enumerate(contexts):
            if ctx.state.finished:
                continue
            if ctx.prefill_started_ns is None:
                ctx.prefill_started_ns = time.time_ns()
            if not ctx.slot_initialized:
                admitted = self.slot_generator.on_admit(ctx)
                if not admitted:
                    continue
            prepared.append(ctx)
        if prepared:
            batch_counts = self.slot_generator.prefill_tokens_multi(prepared, chunk_len)
            prep_iter = iter(batch_counts)
            for idx, ctx in enumerate(contexts):
                if ctx not in prepared:
                    continue
                count = next(prep_iter, 0)
                processed[idx] = count
                if (
                    count > 0
                    and ctx.state.remaining_prompt_tokens == 0
                    and ctx.prefill_completed_ns is None
                ):
                    ctx.prefill_completed_ns = time.time_ns()
        duration = time.perf_counter() - tick_start
        total_tokens = sum(max(val, 0) for val in processed)
        self._prefill_calls += 1
        self._prefill_tokens += total_tokens
        self._prefill_duration_s += max(duration, 0.0)
        return processed

    def decode(self, contexts: Iterable[SequenceContext]) -> Dict[str, float]:
        if self.use_legacy_generator:
            return self._legacy_decode(contexts)

        contexts_list = list(contexts)
        self._last_active_count = len(
            [ctx for ctx in contexts_list if not ctx.state.finished]
        )
        total_tokens = 0
        total_iterations = 0
        total_duration = 0.0
        total_model_duration = 0.0
        total_vectorized = 0
        total_kernel_duration = 0.0
        decode_state: Optional[dict] = None
        if (
            not self.use_legacy_generator
            and self.slot_generator.paged_cache is not None
            and self.decode_unroll > 1
        ):
            decode_state = {"view": None, "seq_ids": ()}

        def emit_callback(active, logits):
            nonlocal total_tokens, total_vectorized
            logits_mx = (
                logits if isinstance(logits, _MX_ARRAY_TYPE) else mx.array(logits)
            )
            if logits_mx.ndim == 1:
                logits_mx = logits_mx[None, :]
            emitted, vectorized = self._sample_tokens_from_logits(active, logits_mx)
            total_tokens += emitted
            total_vectorized += vectorized
            return emitted, vectorized

        chunk_stats = self.slot_generator.decode_chunk(
            contexts_list,
            max_steps=self.decode_unroll,
            decode_state=decode_state,
            safe_bump=self.decode_unroll_safe,
            emit_callback=None if self.use_legacy_generator else emit_callback,
        )
        total_iterations = chunk_stats.get("iterations", 0)
        total_model_duration = chunk_stats.get("model_s", 0.0)
        total_kernel_duration = chunk_stats.get("kernel_s", 0.0)
        total_duration = chunk_stats.get("total_s", 0.0)

        stats = {
            "decode_iterations": total_iterations,
            "decode_tokens": total_tokens,
            "decode_duration_s": max(total_duration, 0.0),
            "prefill_calls": self._prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": self._prefill_duration_s,
        }
        if hasattr(self.slot_generator, "compile_stats"):
            stats.update(self.slot_generator.compile_stats())
        if not self.use_legacy_generator:
            profile = getattr(self.slot_generator, "_last_decode_profile", None) or {}
            stats["decode_model_duration_s"] = total_model_duration
            stats["decode_kernel_duration_s"] = total_kernel_duration
            stats["decode_total_duration_s"] = total_duration
            if total_tokens > 0:
                stats["decode_model_per_token_s"] = total_model_duration / total_tokens
                stats["decode_kernel_per_token_s"] = (
                    total_kernel_duration / total_tokens
                )
                stats["decode_total_per_token_s"] = total_duration / total_tokens
            stats["decode_profile_batch"] = profile.get(
                "batch_size",
                self._last_active_count,
            )
            stats["decode_vectorized"] = total_vectorized
        stats.update(self.slot_generator.prefill_slice_stats())
        self._last_decode_stats = stats
        return stats

    def _sample_tokens_from_logits(
        self, active: List[SequenceContext], logits_mx
    ) -> Tuple[int, int]:
        now = time.time_ns()
        simple_indices: List[int] = []
        for idx, ctx in enumerate(active):
            if ctx.state.finished:
                continue
            if ctx.logits_processors:
                continue
            if not getattr(ctx, "uses_default_sampler", False):
                continue
            simple_indices.append(idx)
        simple_token_map: Dict[int, int] = {}
        if simple_indices:
            simple_logits = mx.array(logits_mx[simple_indices])
            simple_argmax = mx.argmax(simple_logits, axis=-1)
            mx.eval(simple_argmax)
            tokens = simple_argmax.tolist()
            for idx, token in zip(simple_indices, tokens):
                simple_token_map[idx] = int(token)

        emitted_tokens = 0
        for index, ctx in enumerate(active):
            if ctx.state.finished:
                continue
            if index in simple_token_map:
                token_id = simple_token_map[index]
            else:
                row_mx = logits_mx[index : index + 1]
                if ctx.logits_processors:
                    history_mx = mx.array(ctx.history_tokens, dtype=mx.int32)[None, :]
                    for processor in ctx.logits_processors:
                        row_mx = processor(history_mx, row_mx)
                sampler = getattr(ctx, "sampler", None) or self.default_sampler
                selection = sampler(row_mx)
                token_id = self._extract_token(selection)

            ctx.state.generated_tokens += 1
            emitted_tokens += 1
            ctx.history_tokens.append(token_id)
            ctx.last_token_id = token_id
            if ctx.decode_started_ns is None:
                ctx.decode_started_ns = now
            if token_id in getattr(ctx, "stop_tokens", ()):
                logging.info(
                    "stop_token seen request=%s token=%s step=%s",
                    ctx.state.request_id,
                    token_id,
                    ctx.state.generated_tokens,
                )

            if self.use_legacy_generator:
                text_segment = self._update_detokenizer(ctx, token_id)
            else:
                text_segment = ""

            ctx.enqueue_event(
                self._build_response(
                    ctx,
                    token_id,
                    logprobs=None,
                    text=text_segment,
                    finish_reason=None,
                )
            )

            finish_reason = self._evaluate_stop_conditions(ctx, token_id)
            if finish_reason:
                ctx.state.finished = True
                if self.use_legacy_generator and hasattr(ctx, "detokenizer"):
                    ctx.detokenizer.finalize()
                    final_segment = ctx.detokenizer.last_segment or ""
                else:
                    final_segment = ""
                logging.info(
                    "finish_reason=%s req=%s tokens=%s last_ids=%s",
                    finish_reason,
                    ctx.state.request_id,
                    ctx.state.generated_tokens,
                    ctx.history_tokens[-5:],
                )
                ctx.enqueue_event(
                    self._build_response(
                        ctx,
                        token_id,
                        logprobs=None,
                        text=final_segment,
                        finish_reason=finish_reason,
                    )
                )
                self.slot_generator.on_release(ctx)
                if ctx.state.slot_id is not None:
                    self.slot_batcher.release(ctx.state.request_id)

        return emitted_tokens, len(simple_token_map)

    def _legacy_decode(self, contexts: Iterable[SequenceContext]) -> Dict[str, float]:
        start = time.perf_counter()
        responses = self.legacy_generator.next()
        duration = time.perf_counter() - start
        decode_iters = 1 if responses else 0
        emitted_tokens = 0

        now = time.time_ns()
        missing_uids = []
        active_batch = getattr(self.legacy_generator, "active_batch", None)
        if active_batch is not None and hasattr(active_batch, "uids"):
            self._last_active_count = len(active_batch.uids)
        else:
            self._last_active_count = 0

        for response in responses or ():
            ctx = self.uid_to_context.get(response.uid)
            if ctx is None or ctx.state.finished:
                if ctx is None:
                    missing_uids.append(response.uid)
                continue

            token_id = int(response.token)
            logprobs = response.logprobs

            ctx.state.generated_tokens += 1
            emitted_tokens += 1
            ctx.history_tokens.append(token_id)
            ctx.last_token_id = token_id
            if ctx.decode_started_ns is None:
                ctx.decode_started_ns = now

            text_segment = self._update_detokenizer(ctx, token_id)

            if ctx.logits_processors:
                history = mx.array(ctx.history_tokens, dtype=mx.int32)[None, :]
                for processor in ctx.logits_processors:
                    logprobs = processor(history, logprobs)

            ctx.enqueue_event(
                self._build_response(
                    ctx,
                    token_id,
                    logprobs,
                    text_segment,
                    finish_reason=None,
                )
            )

            finish_reason = response.finish_reason
            if not finish_reason:
                finish_reason = self._evaluate_stop_conditions(ctx, token_id)

                if finish_reason:
                    ctx.state.finished = True
                    ctx.detokenizer.finalize()
                    final_segment = ctx.detokenizer.last_segment or ""
                    ctx.enqueue_event(
                        self._build_response(
                            ctx,
                            token_id,
                            logprobs,
                            final_segment,
                            finish_reason=finish_reason,
                        )
                    )
                    self.uid_to_context.pop(response.uid, None)
                    if ctx.state.slot_id is not None:
                        self.slot_batcher.release(ctx.state.request_id)

        if missing_uids:
            logging.warning("decode missing contexts for uids=%s", missing_uids)

        stats = {
            "decode_iterations": decode_iters,
            "decode_tokens": emitted_tokens,
            "decode_duration_s": max(duration, 0.0),
            "prefill_calls": self._prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": self._prefill_duration_s,
            "prefill_hybrid_dense_tokens": 0.0,
            "prefill_hybrid_dense_ms": 0.0,
        }
        if self.prefix_cache is not None:
            prefix_stats = self.prefix_cache.stats()
            for key, value in prefix_stats.items():
                stats[f"prefix_{key}"] = value
        self._last_decode_stats = stats
        return stats

    def debug_state(self) -> Dict[str, int]:
        if self.use_legacy_generator:
            active = getattr(self.legacy_generator, "active_batch", None)
            batch_size = (
                len(active.uids)
                if active is not None and hasattr(active, "uids")
                else 0
            )
        else:
            batch_size = self._last_active_count
        return {
            "generator_active": batch_size,
            "uid_count": len(self.uid_to_context),
            "prefill_calls": self._prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "decode_iterations": self._last_decode_stats.get("decode_iterations", 0),
            "decode_tokens": self._last_decode_stats.get("decode_tokens", 0),
        }

    def collect_step_stats(self) -> Dict[str, float]:
        stats = dict(self._last_decode_stats)
        stats.update(self.slot_generator.prefill_slice_stats())
        stats.setdefault("prefill_calls", self._prefill_calls)
        stats.setdefault("prefill_tokens", self._prefill_tokens)
        stats.setdefault("prefill_duration_s", self._prefill_duration_s)
        stats["kv_pool_blocks"] = float(self.kv_pool_blocks or 0)
        if self.kv_pool_meta:
            for key, value in self.kv_pool_meta.items():
                if isinstance(value, (int, float)):
                    stats[f"kv_pool_{key}"] = float(value)
                else:
                    stats[f"kv_pool_{key}"] = value
        if self.prefix_cache is not None:
            prefix_stats = self.prefix_cache.stats()
            for key, value in prefix_stats.items():
                stats[f"prefix_{key}"] = value
        stats.update(self._memory_stats())
        return stats

    def _estimate_block_bytes(self, geometry: dict) -> int:
        num_layers = geometry.get("num_layers")
        num_kv_heads = geometry.get("num_kv_heads")
        head_dim = geometry.get("head_dim")
        dtype = geometry.get("dtype")
        if not all(
            isinstance(val, int) and val > 0
            for val in (num_layers, num_kv_heads, head_dim)
        ):
            return 0
        dtype_bytes = _dtype_nbytes(dtype)
        tokens_per_block = self.kv_block_size
        base_bytes = (
            num_layers * num_kv_heads * tokens_per_block * head_dim * dtype_bytes * 2
        )
        if self.kv_quant_mode != "int4_v":
            return base_bytes
        group_size = max(1, int(self.kv_quant_group_size))
        bits = _quant_bits_for_mode(self.kv_quant_mode)
        if bits == 0:
            return base_bytes
        bytes_per_group = _bytes_per_quant_group(bits, group_size)
        groups_per_head = _ceil_div(head_dim, group_size)
        per_token_quant = bytes_per_group + 4  # scale + zero (float16 each)
        quant_bytes = (
            num_layers
            * num_kv_heads
            * tokens_per_block
            * groups_per_head
            * per_token_quant
        )
        return base_bytes + quant_bytes

    def _auto_select_kv_pool_blocks(
        self, geometry: dict
    ) -> tuple[Optional[int], Dict[str, float]]:
        metal = getattr(mx, "metal", None)
        if metal is None or not hasattr(metal, "device_info"):
            return None, {"auto_reason": "no_device_info"}
        info = metal.device_info()
        limit = info.get("max_recommended_working_set_size") or info.get(
            "recommendedMaxWorkingSetSize"
        )
        if not limit:
            return None, {"auto_reason": "missing_limit"}
        per_block = self._estimate_block_bytes(geometry)
        if per_block <= 0:
            return None, {"auto_reason": "invalid_geometry"}
        target_bytes = int(_AUTO_KV_POOL_FRACTION * limit)
        blocks = max(1, target_bytes // per_block)
        max_blocks = max(1, _AUTO_KV_POOL_MAX_BLOCKS)
        clamped = False
        if blocks > max_blocks:
            blocks = max_blocks
            clamped = True
        meta = {
            "device_name": info.get("device_name") or info.get("architecture"),
            "device_limit_bytes": float(limit),
            "target_bytes": float(target_bytes),
            "per_block_bytes": float(per_block),
            "auto_blocks": float(blocks),
        }
        if clamped:
            meta["auto_clamped_to"] = float(max_blocks)
        return blocks, meta

    def _memory_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        get_active = getattr(mx, "get_active_memory", None)
        get_peak = getattr(mx, "get_peak_memory", None)
        get_cache = getattr(mx, "get_cache_memory", None)
        active = float(get_active()) if callable(get_active) else 0.0
        peak = float(get_peak()) if callable(get_peak) else 0.0
        cache = float(get_cache()) if callable(get_cache) else 0.0
        stats["memory_active_bytes"] = active
        stats["memory_peak_bytes"] = peak
        stats["memory_cache_bytes"] = cache

        limit = None
        if self.kv_pool_meta:
            limit = self.kv_pool_meta.get("device_limit_bytes")
            if limit is not None:
                limit = float(limit)
        if limit is None:
            metal = getattr(mx, "metal", None)
            if metal is not None and hasattr(metal, "device_info"):
                try:
                    info = metal.device_info()
                    raw_limit = info.get("max_recommended_working_set_size")
                    if raw_limit:
                        limit = float(raw_limit)
                except Exception:  # pragma: no cover - defensive
                    limit = None

        if limit:
            stats["memory_limit_bytes"] = limit
            utilization = active / limit if limit > 0 else 0.0
            stats["memory_utilization"] = utilization
            self._maybe_warn_memory(limit, active, utilization)
        return stats

    def _maybe_warn_memory(
        self, limit: float, active: float, utilization: float
    ) -> None:
        if limit <= 0:
            return
        if utilization >= _MEMORY_WARN_FRACTION and not self._memory_warning_emitted:
            logging.warning(
                "Active memory %.2f MB exceeds %.0f%% of recommended working set (%.2f MB)",
                active / (1024 * 1024),
                _MEMORY_WARN_FRACTION * 100,
                limit / (1024 * 1024),
            )
            self._memory_warning_emitted = True
        elif utilization <= _MEMORY_RESET_FRACTION:
            self._memory_warning_emitted = False

    # ------------------------------------------------------------------#
    # Paged attention scaffolding
    # ------------------------------------------------------------------#
    def _initialize_paged_backend(self) -> None:
        if KVBlockManager is None:
            logging.warning(
                "Paged attention requested but mlx.nn.paged_kv is unavailable; using dense backend."
            )
            return

        geometry = _infer_attention_geometry(self.model)
        if not geometry:
            logging.warning(
                "Unable to infer model geometry for paged attention; using dense backend."
            )
            return

        geometry = dict(geometry)
        manager_dtype = geometry.get("dtype") or mx.float16
        if manager_dtype not in (mx.float16, mx.bfloat16, mx.float32):
            logging.info(
                "Paged KV enforcing dtype=float16 (model dtype %s unsupported)",
                manager_dtype,
            )
            manager_dtype = mx.float16
        geometry["dtype"] = manager_dtype

        blocks_meta: Dict[str, float] = {}
        if self.requested_kv_pool_blocks is None:
            auto_blocks, blocks_meta = self._auto_select_kv_pool_blocks(geometry)
            if auto_blocks is None:
                auto_blocks = _DEFAULT_KV_POOL_BLOCKS
                blocks_meta.setdefault("auto_reason", "fallback_default")
            self.kv_pool_blocks = max(1, int(auto_blocks))
        else:
            self.kv_pool_blocks = max(1, int(self.requested_kv_pool_blocks))
        self.kv_pool_meta = dict(blocks_meta)
        if blocks_meta:
            per_block = blocks_meta.get("per_block_bytes")
            target_bytes = blocks_meta.get("target_bytes")
            device_name = blocks_meta.get("device_name")
            logging.info(
                "Auto KV pool sizing blocks=%s target=%.2f MB per_block=%.2f MB device=%s",
                self.kv_pool_blocks,
                (target_bytes or 0) / (1024 * 1024),
                (per_block or 0) / (1024 * 1024),
                device_name or "unknown",
            )

        try:
            quant_spec = self._build_quant_spec()
            self.kv_manager = KVBlockManager(
                num_layers=geometry["num_layers"],
                num_kv_heads=geometry["num_kv_heads"],
                head_dim=geometry["head_dim"],
                block_size=self.kv_block_size,
                max_blocks=self.kv_pool_blocks,
                dtype=manager_dtype,
                kv_quantization=quant_spec,
            )
        except Exception as exc:  # pragma: no cover - integration path
            logging.warning(
                "Failed to initialize KVBlockManager (disabling paged backend): %s", exc
            )
            self.kv_manager = None
            return

        self.kv_geometry = geometry
        self.kv_head_mapping = geometry.get("kv_head_mapping")
        self.kv_dtype = manager_dtype
        self.paged_backend_enabled = True
        self.actual_attn_backend = "paged"
        backend_signature = BackendSignature(
            block_size=self.kv_block_size,
            vec_width=self.paged_vec_width,
            threads_per_head=self.paged_threads_per_head,
            kv_quant_mode=self.kv_quant_mode,
            model_signature=self._model_signature,
        )
        self.paged_cache = PagedSlotKVCache(
            self.kv_manager,
            max_active=self.total_slots,
            backend_signature=backend_signature,
        )
        self.prefix_cache = PrefixCache(
            self.kv_manager,
            block_size=self.kv_block_size,
            max_entries=self.max_num_seqs * 32,
            reuse_callback=self.paged_cache.mark_prefix_reuse,
        )
        self._prewarm_paged_attention()
        install_paged_attention_patch()
        wrap_attention_layers(self.model)
        if self.slot_generator is not None and not self.use_legacy_generator:
            self.slot_generator.set_paged_backend(
                self.paged_cache,
                self.kv_head_mapping,
                manager=self.kv_manager,
                num_layers=self.kv_geometry.get("num_layers"),
                prefix_cache=self.prefix_cache,
                model_signature=self._model_signature,
                kv_quant_mode=self.kv_quant_mode,
            )

    def _prewarm_paged_attention(self) -> None:
        prewarm = getattr(mx.fast, "_paged_attention_prewarm", None)
        if not prewarm or not self.kv_dtype:
            return
        kwargs = {}
        if self.paged_threads_per_head is not None:
            kwargs["threads_per_head"] = int(self.paged_threads_per_head)
        if self.paged_vec_width is not None:
            kwargs["vec_width"] = int(self.paged_vec_width)
        try:
            prewarm(self.kv_block_size, self.kv_dtype, **kwargs)
        except Exception as exc:  # pragma: no cover - logging only
            logging.warning("Paged attention prewarm failed: %s", exc)

    def _compute_apc_prefixes(
        self, tokens: Sequence[int]
    ) -> Optional[List[Tuple[int, bytes]]]:
        if not tokens or not self.paged_backend_enabled or self.kv_block_size <= 0:
            return None
        block = int(self.kv_block_size)
        if len(tokens) < block:
            return None
        prefixes: List[Tuple[int, bytes]] = []
        for seq_len in range(block, len(tokens) + 1, block):
            chunk = tokens[:seq_len]
            digest = hashlib.sha1(",".join(map(str, chunk)).encode("utf-8")).digest()
            prefixes.append((seq_len, digest))
        return prefixes or None

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def _extract_token(self, value) -> int:
        if hasattr(value, "item"):
            try:
                safe_eval(value)
                return int(value.item())
            except (TypeError, ValueError, RuntimeError):
                pass
        if hasattr(value, "tolist"):
            try:
                safe_eval(value)
            except Exception:
                pass
            try:
                data = value.tolist()
            except RuntimeError:
                data = None
            if isinstance(data, list):
                while data and isinstance(data[0], list):
                    data = data[0]
                if data:
                    return int(data[0])
        if isinstance(value, (list, tuple)):
            data = value
            while data and isinstance(data[0], (list, tuple)):
                data = data[0]
            if data:
                return int(data[0])
        try:
            return int(value)
        except RuntimeError:
            fallback = None
            if hasattr(value, "tolist"):
                try:
                    fallback = value.tolist()
                except RuntimeError:
                    fallback = None
            if isinstance(fallback, list):
                data = fallback
                while data and isinstance(data[0], list):
                    data = data[0]
                if data:
                    return int(data[0])
            logging.debug(
                "extract_token fallback defaulting to 0 for %s", type(value).__name__
            )
            return 0

    def _batched_sampler(self, logprobs: mx.array) -> mx.array:
        batch = getattr(self.legacy_generator, "active_batch", None)
        if batch is None or not batch.uids:
            return mx.argmax(logprobs, axis=-1)

        if logprobs.ndim == 0 or logprobs.shape[-1] == 0:
            return mx.zeros((logprobs.shape[0],), dtype=mx.uint32)

        picks: List[mx.array] = []
        rows = int(logprobs.shape[0])
        vocab = logprobs.shape[-1] if logprobs.ndim >= 2 else 0
        for idx, uid in enumerate(batch.uids):
            if idx >= rows or vocab == 0:
                logging.warning(
                    "missing logprobs row=%s rows=%s vocab=%s", idx, rows, vocab
                )
                picks.append(mx.zeros((1,), dtype=mx.uint32))
                continue
            ctx = self.uid_to_context.get(uid)
            sampler = getattr(ctx, "sampler", None) or self.default_sampler
            selection = sampler(logprobs[idx : idx + 1])
            picks.append(selection)
        return mx.concatenate(picks, axis=0)

    def _update_detokenizer(self, ctx: SequenceContext, token_id: int) -> str:
        ctx.detokenizer.add_token(token_id)
        return ctx.detokenizer.last_segment

    def _evaluate_stop_conditions(
        self, ctx: SequenceContext, token_id: int
    ) -> Optional[str]:
        if getattr(ctx, "stop_tokens", None) and token_id in ctx.stop_tokens:
            return "stop"
        for stop_seq in ctx.stop_sequences:
            if stop_seq and ctx.history_tokens[-len(stop_seq) :] == stop_seq:
                return "stop"
        if ctx.state.generated_tokens >= ctx.state.max_new_tokens:
            return "length"
        return None

    def _build_response(
        self,
        ctx: SequenceContext,
        token_id: int,
        logprobs: Optional[mx.array],
        text: str,
        finish_reason: Optional[str],
    ) -> GenerationResponse:
        prompt_tokens = ctx.state.prompt_len
        prompt_tps = 0.0
        if (
            ctx.prefill_started_ns is not None
            and ctx.prefill_completed_ns is not None
            and ctx.prefill_completed_ns > ctx.prefill_started_ns
        ):
            elapsed = (ctx.prefill_completed_ns - ctx.prefill_started_ns) / 1e9
            if elapsed > 0:
                prompt_tps = prompt_tokens / elapsed

        generation_tokens = ctx.state.generated_tokens
        generation_tps = 0.0
        if ctx.decode_started_ns is not None:
            elapsed = (time.time_ns() - ctx.decode_started_ns) / 1e9
            if elapsed > 0:
                generation_tps = generation_tokens / elapsed

        return GenerationResponse(
            text=text,
            token=token_id,
            logprobs=logprobs,
            from_draft=False,
            prompt_tokens=prompt_tokens,
            prompt_tps=prompt_tps,
            generation_tokens=generation_tokens,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason,
        )


def _walk_model_variants(root) -> Iterable[object]:
    seen = set()
    queue = [root]
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for attr in _MODEL_CHILD_ATTRS:
            child = getattr(obj, attr, None)
            if child is None:
                continue
            queue.append(child)


def _find_layers_list(model) -> Optional[Sequence]:
    for candidate in _walk_model_variants(model):
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, (list, tuple)) and layers:
            return layers
    return None


def _find_model_args(model):
    for candidate in _walk_model_variants(model):
        args = getattr(candidate, "args", None)
        if args is not None:
            return args
    return None


def _locate_attention_module(layer):
    for attr in ("self_attn", "attention", "attn", "self_attention"):
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None


def _infer_model_dtype(model):
    for candidate in _walk_model_variants(model):
        embed = getattr(candidate, "embed_tokens", None)
        if embed is None:
            continue
        weight = getattr(embed, "weight", None)
        if weight is not None and hasattr(weight, "dtype"):
            return weight.dtype
    return mx.float16


def _build_head_mapping(
    num_heads: Optional[int], num_kv_heads: Optional[int]
) -> Optional[mx.array]:
    if not num_heads or not num_kv_heads or num_heads == num_kv_heads:
        return None
    mapping = [(head * num_kv_heads) // num_heads for head in range(num_heads)]
    return mx.array(mapping, dtype=mx.int32)


def _infer_attention_geometry(model) -> Optional[dict]:
    layers = _find_layers_list(model)
    args = _find_model_args(model)
    num_layers = getattr(args, "num_hidden_layers", None)
    if num_layers is None and layers:
        num_layers = len(layers)

    attn = _locate_attention_module(layers[0]) if layers else None
    num_heads = None
    if attn is not None:
        num_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_heads", None)
    if num_heads is None and args is not None:
        num_heads = getattr(args, "num_attention_heads", None)

    num_kv_heads = None
    if attn is not None:
        num_kv_heads = (
            getattr(attn, "n_kv_heads", None)
            or getattr(attn, "num_kv_heads", None)
            or getattr(attn, "num_key_value_heads", None)
        )
    if num_kv_heads is None and args is not None:
        num_kv_heads = getattr(args, "num_key_value_heads", None)

    head_dim = getattr(attn, "head_dim", None) if attn is not None else None
    if head_dim is None and args is not None:
        head_dim = getattr(args, "head_dim", None)
    if head_dim is None and args is not None:
        hidden = getattr(args, "hidden_size", None)
        if hidden and num_heads:
            head_dim = hidden // num_heads

    dtype = _infer_model_dtype(model)

    if not all((num_layers, num_kv_heads, head_dim)):
        return None

    return {
        "num_layers": int(num_layers),
        "num_heads": int(num_heads) if num_heads else None,
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "dtype": dtype,
        "kv_head_mapping": _build_head_mapping(num_heads, num_kv_heads),
    }


def _dtype_nbytes(dtype) -> int:
    if dtype in (mx.float16, mx.bfloat16):
        return 2
    if dtype == mx.float32:
        return 4
    return 2


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _bytes_per_quant_group(bits: int, group_size: int) -> int:
    return math.ceil(group_size * bits / 8)


def _quant_bits_for_mode(mode: str) -> int:
    if mode == "int4_v":
        return 4
    return 0
