# ABOUTME: Coordinates scheduler loop and request streaming for continuous batching.
# ABOUTME: Bridges synchronous HTTP handlers with background worker threads.

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Iterable, Optional, Sequence, Tuple

import mlx.core as mx

from ..generate import GenerationResponse
from .engine import ModelRunner
from .scheduler import Scheduler
from .state import SequenceContext


class ContinuousBatchingRuntime:
    """Owns lifecycle of the scheduler and exposes request-level generators."""

    def __init__(
        self,
        runner: ModelRunner,
        *,
        max_num_seqs: int,
        max_tokens_per_step: int,
        prefill_chunk: int,
        debug_metrics: bool = False,
    ):
        self.runner = runner
        self.scheduler = Scheduler(
            runner=runner,
            max_num_seqs=max_num_seqs,
            max_tokens_per_step=max_tokens_per_step,
            prefill_chunk=prefill_chunk,
        )
        self._debug_metrics = debug_metrics
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name="mlx-lm-batcher", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------#
    # Worker loop
    # ------------------------------------------------------------------#
    def _worker_loop(self) -> None:
        backoff = 0.002
        while not self._shutdown.is_set():
            if self.scheduler.has_pending_work:
                self.scheduler.step()
                metrics = self.scheduler.metrics
                if self._debug_metrics:
                    runner_state = self.runner.debug_state()
                    logging.info(
                        "tick wait=%s active=%s prefill_tokens=%s prefill_calls=%s prefill_ms=%.2f decode_batch=%s decode_iters=%s decode_tokens=%s decode_ms=%.2f generator_active=%s uid_map=%s",
                        metrics["wait_queue_depth"],
                        metrics["active_sequences"],
                        metrics.get("prefill_tokens", 0),
                        metrics.get("prefill_calls", 0),
                        metrics.get("prefill_duration_s", 0.0) * 1000.0,
                        metrics.get("decode_batch_size", 0),
                        metrics.get("decode_iterations", 0),
                        metrics.get("decode_tokens", 0),
                        metrics.get("decode_duration_s", 0.0) * 1000.0,
                        runner_state.get("generator_active", 0),
                        runner_state.get("uid_count", 0),
                    )
                else:
                    logging.debug(
                        "scheduler_tick inflight=%s active=%s prefill_tokens=%s decode_batch=%s",
                        metrics["wait_queue_depth"] + metrics["active_sequences"],
                        metrics["active_sequences"],
                        metrics.get("prefill_tokens", 0),
                        metrics.get("decode_batch_size", 0),
                    )
                backoff = 0.002
            else:
                woke = self._wake.wait(backoff)
                if woke:
                    self._wake.clear()
                backoff = min(backoff * 1.5, 0.05)

    def shutdown(self) -> None:
        self.scheduler.stop()
        self._shutdown.set()
        self._wake.set()
        self._worker.join(timeout=1.0)

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def submit_request(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_new_tokens: int,
        sampler_settings: Dict[str, float],
        stopping_settings: Dict[str, Optional[int]],
        logit_bias: Optional[Dict[int, float]] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
    ) -> Tuple[str, Iterable[GenerationResponse]]:
        request_id = str(uuid.uuid4())
        ctx = self.runner.build_context(
            request_id,
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            sampler_settings=sampler_settings,
            stopping_settings=stopping_settings,
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        self.scheduler.enqueue(ctx)
        self._wake.set()
        return request_id, self._stream(ctx)

    def _stream(self, ctx: SequenceContext):
        while True:
            if ctx.state.cancel_requested and ctx.events.empty():
                yield self._cancel_event(ctx)
                break

            event = ctx.events.get()
            yield event
            if getattr(event, "finish_reason", None):
                break

    def _cancel_event(self, ctx: SequenceContext) -> GenerationResponse:
        prompt_tokens = ctx.state.prompt_len
        prompt_tps = 0.0
        generation_tokens = ctx.state.generated_tokens
        generation_tps = 0.0
        if ctx.decode_started_ns:
            elapsed = (time.time_ns() - ctx.decode_started_ns) / 1e9
            if elapsed > 0:
                generation_tps = generation_tokens / elapsed
        vocab_size = getattr(self.runner.tokenizer, 'vocab_size', 1) or 1
        return GenerationResponse(
            text="",
            token=ctx.history_tokens[-1] if ctx.history_tokens else 0,
            logprobs=mx.zeros((1, vocab_size)),
            from_draft=False,
            prompt_tokens=prompt_tokens,
            prompt_tps=prompt_tps,
            generation_tokens=generation_tokens,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="cancelled",
        )


def create_runtime(
    config: Dict[str, int],
    *,
    model,
    tokenizer,
    draft_model=None,
) -> Optional[ContinuousBatchingRuntime]:
    if not config.get("enabled"):
        return None
    if model is None or tokenizer is None:
        return None

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        max_num_seqs=config["max_num_seqs"],
        prefill_chunk=config["prefill_chunk"],
    )
    return ContinuousBatchingRuntime(
        runner=runner,
        max_num_seqs=config["max_num_seqs"],
        max_tokens_per_step=config["max_tokens_per_step"],
        prefill_chunk=config["prefill_chunk"],
    )


__all__ = ["ContinuousBatchingRuntime", "create_runtime"]
