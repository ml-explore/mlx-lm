# ABOUTME: Implements continuous batching engine bridging slot generator modes.
# ABOUTME: Provides ModelRunner wrapper that maps model outputs to contexts.

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from ..generate import BatchGenerator, GenerationResponse
from ..sample_utils import make_logits_processors, make_sampler
from .state import SequenceContext, SequenceState
from .slot_allocator import SlotAllocator
from .slot_batcher import SlotBatcher
from .slot_generator import SlotGenerator
from .slot_kv_cache import SlotKVCache


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
        force_legacy_generator: bool = False,
        stop_tokens: Optional[Iterable[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.max_num_seqs = max_num_seqs
        self.prefill_chunk = prefill_chunk
        self.use_legacy_generator = force_legacy_generator
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
        xtc_special = list(self.stop_tokens) if self.stop_tokens else ([eos_id] if eos_id is not None else [])

        self.default_sampler = make_sampler(
            temp=0.0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            xtc_special_tokens=xtc_special,
        )

        # Slot scaffolding for upcoming vectorized decode.
        self.slot_allocator = SlotAllocator(max_num_seqs)
        # Placeholder KV cache capacity equal to prefill chunk for stage scaffolding.
        self.slot_kv_cache = SlotKVCache(
            max_slots=max_num_seqs,
            capacity_tokens=prefill_chunk or 1,
            kv_heads=1,
            head_dim=1,
        )
        self.slot_batcher = SlotBatcher(self.slot_allocator, self.slot_kv_cache)
        self.uid_to_context: Dict[int, SequenceContext] = {}

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
            )

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
        if self.use_legacy_generator:
            ctx.detokenizer = self.tokenizer.detokenizer
        if prompt_tokens:
            ctx.last_token_id = prompt_tokens[-1]
        else:
            ctx.last_token_id = getattr(self.tokenizer, "bos_token_id", self.tokenizer.eos_token_id)
        ctx.history_tokens = list(prompt_tokens)
        ctx.stop_sequences = list(stopping_cfg.get("stop_id_sequences", []))
        ctx.stop_tokens = set(self.stop_tokens)
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

    def decode(self, contexts: Iterable[SequenceContext]) -> Dict[str, float]:
        if self.use_legacy_generator:
            return self._legacy_decode(contexts)

        active = [ctx for ctx in contexts if not ctx.state.finished]
        start = time.perf_counter()
        logits = self.slot_generator.decode_step(active) if active else mx.zeros((0, 0), dtype=mx.float32)
        duration = time.perf_counter() - start

        logits_mx = mx.array(logits)
        if logits_mx.ndim == 1:
            logits_mx = logits_mx[None, :]

        emitted_tokens = 0
        now = time.time_ns()
        self._last_active_count = len(active)

        for index, ctx in enumerate(active):
            if ctx.state.finished:
                continue
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

        stats = {
            "decode_iterations": 1 if emitted_tokens else 0,
            "decode_tokens": emitted_tokens,
            "decode_duration_s": max(duration, 0.0),
            "prefill_calls": self._prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": self._prefill_duration_s,
        }
        self._last_decode_stats = stats
        return stats

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
        }
        self._last_decode_stats = stats
        return stats

    def debug_state(self) -> Dict[str, int]:
        if self.use_legacy_generator:
            active = getattr(self.legacy_generator, "active_batch", None)
            batch_size = len(active.uids) if active is not None and hasattr(active, "uids") else 0
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
        stats.setdefault("prefill_calls", self._prefill_calls)
        stats.setdefault("prefill_tokens", self._prefill_tokens)
        stats.setdefault("prefill_duration_s", self._prefill_duration_s)
        return stats

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def _extract_token(self, value) -> int:
        if hasattr(value, "item"):
            try:
                mx.eval(value)
                return int(value.item())
            except (TypeError, ValueError):
                pass
        if hasattr(value, "tolist"):
            try:
                mx.eval(value)
            except Exception:
                pass
            data = value.tolist()
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
        return int(value)

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

    def _evaluate_stop_conditions(self, ctx: SequenceContext, token_id: int) -> Optional[str]:
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
