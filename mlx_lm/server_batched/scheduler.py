# ABOUTME: Implements iteration-level scheduler for continuous batching.
# ABOUTME: Manages prefill and decode stages with queue-based admission.

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Deque, List

from .state import SequenceContext


class Scheduler:
    def __init__(
        self,
        runner,
        max_num_seqs: int,
        max_tokens_per_step: int,
        prefill_chunk: int,
    ):
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if max_tokens_per_step <= 0:
            raise ValueError("max_tokens_per_step must be positive")
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")

        self.runner = runner
        self.max_num_seqs = max_num_seqs
        self.max_tokens_per_step = max_tokens_per_step
        self.prefill_chunk = prefill_chunk

        self._wait_queue: Deque[SequenceContext] = deque()
        self._active: List[SequenceContext] = []
        self._lock = threading.Lock()
        self._stop = False
        self._last_prefill_tokens = 0
        self._last_decode_batch = 0
        self._last_step_metrics = self._make_empty_metrics()

    def _make_empty_metrics(self) -> dict:
        return {
            "prefill_calls": 0,
            "prefill_tokens": 0,
            "prefill_duration_s": 0.0,
            "prefill_wall_s": 0.0,
            "prefill_sequences": 0,
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
            "decode_batch_size": 0,
        }

    def enqueue(self, ctx: SequenceContext) -> None:
        with self._lock:
            if self._stop or ctx.state.finished:
                return
            self._wait_queue.append(ctx)

    def step(self) -> None:
        self.runner.begin_step()
        prefill_duration = 0.0
        ready_count = 0
        with self._lock:
            if self._stop:
                self._last_step_metrics = self._make_empty_metrics()
                return
            prefill_start = time.perf_counter()
            ready = self._prefill_until_budget(self.max_tokens_per_step)
            prefill_duration = time.perf_counter() - prefill_start
            ready_count = len(ready)
            if ready:
                self._active.extend(ready)

            if self._stop or not self._active:
                stats = self.runner.collect_step_stats()
                self._last_prefill_tokens = stats.get("prefill_tokens", self._last_prefill_tokens)
                self._last_decode_batch = 0
                self._last_step_metrics = self._compose_metrics(
                    stats=stats,
                    prefill_sequences=ready_count,
                    decode_batch_size=0,
                    prefill_wall=prefill_duration,
                )
                return

            active_snapshot = list(self._active)

        decode_stats = self.runner.decode(active_snapshot)
        self._last_decode_batch = len(active_snapshot)

        with self._lock:
            self._active = [ctx for ctx in self._active if not ctx.state.finished]
            stats = decode_stats or self.runner.collect_step_stats()
            self._last_prefill_tokens = stats.get("prefill_tokens", self._last_prefill_tokens)
            self._last_step_metrics = self._compose_metrics(
                stats=stats,
                prefill_sequences=ready_count,
                decode_batch_size=self._last_decode_batch,
                prefill_wall=prefill_duration,
            )

    def _prefill_until_budget(self, token_budget: int) -> List[SequenceContext]:
        ready: List[SequenceContext] = []
        consumed = 0
        requeue: List[SequenceContext] = []

        while (
            not self._stop
            and self._wait_queue
            and len(self._active) + len(ready) < self.max_num_seqs
        ):
            ctx = self._wait_queue.popleft()
            if ctx.state.finished:
                continue
            remaining = ctx.state.remaining_prompt_tokens
            if remaining <= 0:
                ready.append(ctx)
                continue

            take = min(self.prefill_chunk, remaining, token_budget - consumed)
            if take <= 0:
                requeue.append(ctx)
                break

            self.runner.prefill_context(ctx, take)
            consumed += take
            if ctx.state.remaining_prompt_tokens == 0:
                ready.append(ctx)
            else:
                requeue.append(ctx)

            if consumed >= token_budget:
                break

        for ctx in reversed(requeue):
            self._wait_queue.appendleft(ctx)
        self._last_prefill_tokens = consumed
        return ready

    def _compose_metrics(
        self,
        *,
        stats: dict,
        prefill_sequences: int,
        decode_batch_size: int,
        prefill_wall: float,
    ) -> dict:
        metrics = {
            "prefill_calls": stats.get("prefill_calls", 0),
            "prefill_tokens": stats.get("prefill_tokens", 0),
            "prefill_duration_s": stats.get("prefill_duration_s", 0.0),
            "prefill_wall_s": prefill_wall,
            "prefill_sequences": prefill_sequences,
            "decode_iterations": stats.get("decode_iterations", 0),
            "decode_tokens": stats.get("decode_tokens", 0),
            "decode_duration_s": stats.get("decode_duration_s", 0.0),
            "decode_batch_size": decode_batch_size,
        }
        return metrics

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._wait_queue.clear()
            self._active.clear()

    @property
    def has_pending_work(self) -> bool:
        with self._lock:
            return (not self._stop) and bool(self._wait_queue or self._active)

    @property
    def metrics(self) -> dict:
        with self._lock:
            metrics = dict(self._last_step_metrics)
            metrics.update(
                {
                    "wait_queue_depth": len(self._wait_queue),
                    "active_sequences": len(self._active),
                }
            )
            metrics.setdefault("prefill_tokens", self._last_prefill_tokens)
            metrics["prefill_tokens_step"] = metrics.get("prefill_tokens", 0)
            metrics.setdefault("decode_batch_size", self._last_decode_batch)
            return metrics
