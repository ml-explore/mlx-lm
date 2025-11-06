# ABOUTME: Implements iteration-level scheduler for continuous batching.
# ABOUTME: Manages prefill and decode stages with queue-based admission.

from __future__ import annotations

from collections import deque
import threading
import time
from itertools import islice
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
        self._active: Deque[SequenceContext] = deque()
        self._lock = threading.Lock()
        self._stop = False
        self._last_prefill_tokens = 0
        self._last_decode_batch = 0
        self._last_step_metrics = self._make_empty_metrics()
        runner_buckets = getattr(runner, "decode_buckets", None)
        if runner_buckets:
            self._decode_buckets = list(runner_buckets)
        else:
            self._decode_buckets = self._default_decode_buckets(max_num_seqs)
        self._decode_target = self._decode_buckets[0]

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
        with self._lock:
            if self._stop:
                self._last_step_metrics = self._make_empty_metrics()
                return
            active_count = len(self._active)
            wait_queue_empty = not self._wait_queue
            open_loop_mode = getattr(self.runner, "open_loop_mode", False)
            draining = getattr(self.runner, "open_loop_draining", False)
            need_decode = active_count > 0 and (
                active_count >= self._decode_target
                or (wait_queue_empty and (not open_loop_mode or draining))
            )

            if not need_decode:
                prefill_start = time.perf_counter()
                ready = self._prefill_until_budget(self.max_tokens_per_step)
                prefill_duration = time.perf_counter() - prefill_start
                ready_count = len(ready)
                if ready:
                    self._active.extend(ready)
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

            bucket = self._decode_buckets[-1]
            for candidate in self._decode_buckets:
                if candidate <= active_count:
                    bucket = candidate
                    break
            bucket = max(1, min(bucket, active_count))
            active_snapshot = list(islice(self._active, 0, bucket))

        decode_stats = self.runner.decode(active_snapshot)
        decode_batch = len(active_snapshot)

        with self._lock:
            original_active = list(self._active)
            processed_ids = {id(ctx) for ctx in active_snapshot}
            others = [ctx for ctx in original_active if id(ctx) not in processed_ids and not ctx.state.finished]
            processed_survivors = [ctx for ctx in active_snapshot if not ctx.state.finished]
            self._active = deque(others + processed_survivors)
            stats = decode_stats or self.runner.collect_step_stats()
            self._last_prefill_tokens = stats.get("prefill_tokens", 0)
            self._last_decode_batch = decode_batch
            self._last_step_metrics = self._compose_metrics(
                stats=stats,
                prefill_sequences=0,
                decode_batch_size=self._last_decode_batch,
                prefill_wall=0.0,
            )

    def _prefill_until_budget(self, token_budget: int) -> List[SequenceContext]:
        ready: List[SequenceContext] = []
        consumed = 0
        requeue: List[SequenceContext] = []

        while not self._stop and self._wait_queue and consumed < token_budget:
            ctx = self._wait_queue.popleft()
            if ctx.state.finished:
                continue
            remaining = ctx.state.remaining_prompt_tokens
            if remaining <= 0:
                if len(self._active) + len(ready) < self.max_num_seqs:
                    ready.append(ctx)
                else:
                    requeue.append(ctx)
                continue

            if (
                ctx.state.slot_id is None
                and not getattr(ctx, "slot_initialized", False)
                and len(self._active) + len(ready) >= self.max_num_seqs
            ):
                requeue.append(ctx)
                break

            take = min(self.prefill_chunk, remaining, token_budget - consumed)
            if take <= 0:
                requeue.append(ctx)
                break

            actual = self.runner.prefill_context(ctx, take)
            if actual <= 0:
                requeue.append(ctx)
                break
            consumed += actual
            if ctx.state.remaining_prompt_tokens == 0:
                if len(self._active) + len(ready) < self.max_num_seqs:
                    ready.append(ctx)
                else:
                    requeue.append(ctx)
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
    def _default_decode_buckets(self, max_num_seqs: int) -> List[int]:
        fractions = (1.0, 0.75, 0.5, 0.25, 0.125)
        buckets = {max(1, int(round(max_num_seqs * frac))) for frac in fractions}
        buckets.add(1)
        return sorted(buckets, reverse=True)
