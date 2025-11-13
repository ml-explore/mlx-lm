# ABOUTME: Implements iteration-level scheduler for continuous batching.
# ABOUTME: Manages prefill and decode stages with queue-based admission.

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from itertools import islice
from typing import Deque, List, Optional, Sequence

from .state import SequenceContext

_BENCH_TRACE = os.environ.get("MLXLM_BENCH_TRACE", "").lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}
_BENCH_LOG = logging.getLogger("mlx_lm.bench")


class Scheduler:
    def __init__(
        self,
        runner,
        max_num_seqs: int,
        max_tokens_per_step: int,
        prefill_chunk: int,
        prefill_queue_limit: int = 0,
        *,
        prefill_slice_cap: int | None = None,
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
        self.prefill_slice_cap = max(1, prefill_slice_cap or prefill_chunk)

        self._wait_queue: Deque[SequenceContext] = deque()
        self._active: Deque[SequenceContext] = deque()
        self._ready: Deque[SequenceContext] = deque()
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
        self._ready_limit = max(0, int(prefill_queue_limit))
        self._prefill_batch_fn = getattr(self.runner, "prefill_context_batch", None)
        self._prefill_single_fn = getattr(self.runner, "prefill_context", None)

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

    def _ready_capacity_total(self) -> int:
        return self.max_num_seqs + self._ready_limit

    def _can_accept_ready(self, local_ready_count: int) -> bool:
        total_ready = len(self._active) + len(self._ready) + local_ready_count
        return total_ready < self._ready_capacity_total()

    def _queue_ready_candidate(
        self,
        ctx: SequenceContext,
        ready: List[SequenceContext],
    ) -> bool:
        if getattr(ctx.state, "finished", False):
            return False
        if any(existing is ctx for existing in ready):
            return True
        if any(existing is ctx for existing in self._ready):
            return True
        if any(existing is ctx for existing in self._active):
            return True
        if len(self._active) + len(ready) < self.max_num_seqs:
            ready.append(ctx)
            return True
        if self._can_accept_ready(len(ready)):
            self._ready.append(ctx)
            return True
        return False

    def _promote_ready_locked(self) -> None:
        self._prune_finished_locked()
        while self._ready and len(self._active) < self.max_num_seqs:
            ctx = self._ready.popleft()
            if getattr(ctx.state, "finished", False):
                continue
            self._active.append(ctx)
        self._rebalance_active_locked()

    def _rebalance_active_locked(self) -> None:
        while len(self._active) > self.max_num_seqs:
            ctx = self._active.pop()
            if getattr(ctx.state, "finished", False):
                continue
            self._ready.appendleft(ctx)

    def _prune_finished_locked(self) -> None:
        if not self._active:
            return
        survivors = [
            ctx for ctx in self._active if not getattr(ctx.state, "finished", False)
        ]
        if len(survivors) != len(self._active):
            self._active = deque(survivors)

    def enqueue(self, ctx: SequenceContext) -> None:
        with self._lock:
            if self._stop or ctx.state.finished:
                return
            self._wait_queue.append(ctx)

    def step(self) -> None:
        self.runner.begin_step()
        prefill_wall = 0.0
        prefill_sequences = 0
        ran_prefill = False
        with self._lock:
            self._prune_finished_locked()
            self._promote_ready_locked()
            if self._stop:
                self._last_step_metrics = self._make_empty_metrics()
                return
            active_count = len(self._active)
            had_active = active_count > 0
            wait_queue_empty = not self._wait_queue
            open_loop_mode = getattr(self.runner, "open_loop_mode", False)
            draining = getattr(self.runner, "open_loop_draining", False)
            need_decode = active_count > 0 and (
                active_count >= self._decode_target
                or (wait_queue_empty and (not open_loop_mode or draining))
            )
            if _BENCH_TRACE:
                _BENCH_LOG.info(
                    "scheduler.step need_decode=%s active=%s wait=%s target=%s draining=%s open_loop=%s",
                    need_decode,
                    active_count,
                    len(self._wait_queue),
                    self._decode_target,
                    draining,
                    open_loop_mode,
                )

            if not need_decode:
                prefill_start = time.perf_counter()
                ready = self._prefill_until_budget(
                    self.max_tokens_per_step, skip_ready=True
                )
                prefill_wall = time.perf_counter() - prefill_start
                ready_count = len(ready)
                prefill_sequences = ready_count
                ran_prefill = True
                if ready:
                    self._active.extend(ready)
                    self._rebalance_active_locked()
                stats = self.runner.collect_step_stats()
                self._last_prefill_tokens = stats.get(
                    "prefill_tokens", self._last_prefill_tokens
                )
                self._last_decode_batch = 0
                active_count = len(self._active)
                wait_queue_empty = not self._wait_queue
                bootstrap_decode = (not had_active) and active_count > 0
                need_decode = active_count > 0 and (
                    active_count >= self._decode_target
                    or (wait_queue_empty and (not open_loop_mode or draining))
                    or bootstrap_decode
                )
                self._last_step_metrics = self._compose_metrics(
                    stats=stats,
                    prefill_sequences=ready_count,
                    decode_batch_size=0,
                    prefill_wall=prefill_wall,
                )
                if _BENCH_TRACE:
                    _BENCH_LOG.info(
                        "scheduler.prefill ready=%s consumed=%s wait=%s active=%s duration_ms=%.3f",
                        ready_count,
                        self._last_prefill_tokens,
                        len(self._wait_queue),
                        len(self._active),
                        prefill_wall * 1000.0,
                    )
                if not need_decode:
                    return

            bucket = self._decode_buckets[-1]
            for candidate in self._decode_buckets:
                if candidate <= active_count:
                    bucket = candidate
                    break
            bucket = max(1, min(bucket, active_count))
            active_snapshot = list(islice(self._active, 0, bucket))

        decode_start = time.perf_counter()
        decode_stats = self.runner.decode(active_snapshot)
        decode_wall = time.perf_counter() - decode_start
        decode_batch = len(active_snapshot)

        with self._lock:
            original_active = list(self._active)
            processed_ids = {id(ctx) for ctx in active_snapshot}
            others = [
                ctx
                for ctx in original_active
                if id(ctx) not in processed_ids and not ctx.state.finished
            ]
            processed_survivors = [
                ctx for ctx in active_snapshot if not ctx.state.finished
            ]
            self._active = deque(others + processed_survivors)
            stats = decode_stats or self.runner.collect_step_stats()
            self._last_prefill_tokens = stats.get("prefill_tokens", 0)
            self._last_decode_batch = decode_batch
            self._last_step_metrics = self._compose_metrics(
                stats=stats,
                prefill_sequences=prefill_sequences if ran_prefill else 0,
                decode_batch_size=self._last_decode_batch,
                prefill_wall=prefill_wall if ran_prefill else 0.0,
            )
            if not ran_prefill:
                self._prefill_overlap_locked()
            if _BENCH_TRACE:
                _BENCH_LOG.info(
                    "scheduler.decode batch=%s bucket=%s wait=%s active_after=%s decode_tokens=%s decode_duration_ms=%.3f",
                    decode_batch,
                    bucket,
                    len(self._wait_queue),
                    len(self._active),
                    stats.get("decode_tokens", 0),
                    decode_wall * 1000.0,
                )

    def _prefill_until_budget(
        self, token_budget: int, *, skip_ready: bool = False
    ) -> List[SequenceContext]:
        ready: List[SequenceContext] = []
        consumed = 0
        requeue: List[SequenceContext] = []

        batch: List[tuple[SequenceContext, int]] = []
        total_capacity = self._ready_capacity_total()
        reserved_ready = len(self._active) + len(self._ready)
        max_new_prefills = max(0, total_capacity - reserved_ready)
        decode_target = getattr(self, "_decode_target", self.max_num_seqs)
        if len(self._active) == 0 and max_new_prefills > 0:
            max_new_prefills = min(max_new_prefills, max(1, decode_target))
        new_prefills_started = 0
        while (
            not self._stop
            and self._wait_queue
            and consumed < token_budget
            and self._can_accept_ready(len(ready))
        ):
            ctx = self._wait_queue.popleft()
            if ctx.state.finished:
                continue
            remaining = ctx.state.remaining_prompt_tokens
            if remaining <= 0:
                self._ready.append(ctx)
                continue
            if (
                skip_ready
                and getattr(ctx.state, "prefill_decode_ready", False)
                and remaining > 0
            ):
                self._queue_ready_candidate(ctx, ready)
                requeue.append(ctx)
                continue
            is_new = getattr(ctx, "prefill_started_ns", None) is None
            if is_new and max_new_prefills <= 0:
                requeue.append(ctx)
                continue
            if is_new and new_prefills_started >= max_new_prefills:
                requeue.append(ctx)
                continue
            if (
                ctx.state.slot_id is None
                and not getattr(ctx, "slot_initialized", False)
                and not self._can_accept_ready(len(ready))
            ):
                requeue.append(ctx)
                break
            take = min(
                self.prefill_slice_cap,
                self.prefill_chunk,
                remaining,
                token_budget - consumed,
            )
            if take <= 0:
                requeue.append(ctx)
                break
            batch.append((ctx, take))
            if is_new:
                new_prefills_started += 1
            if max_new_prefills > 0 and new_prefills_started >= max_new_prefills:
                break

        if batch:
            chunk = min(take for _, take in batch)
            contexts = [ctx for ctx, _ in batch]
            produced = self._run_prefill_batch(contexts, chunk)
            if produced is None:
                produced = [chunk] * len(contexts)
            for ctx, actual in zip(contexts, produced):
                consumed += max(actual, 0)
                if actual <= 0:
                    requeue.append(ctx)
                elif ctx.state.remaining_prompt_tokens == 0:
                    if not self._queue_ready_candidate(ctx, ready):
                        requeue.append(ctx)
                else:
                    if getattr(ctx.state, "prefill_decode_ready", False):
                        self._queue_ready_candidate(ctx, ready)
                    requeue.append(ctx)

        for ctx in reversed(requeue):
            self._wait_queue.appendleft(ctx)
        self._last_prefill_tokens = consumed
        if _BENCH_TRACE:
            _BENCH_LOG.info(
                "scheduler.prefill_window consumed=%s ready=%s requeue=%s wait=%s active=%s",
                consumed,
                len(ready),
                len(requeue),
                len(self._wait_queue),
                len(self._active),
            )
        self._promote_ready_locked()
        return ready

    def _run_prefill_batch(self, contexts: Sequence[SequenceContext], chunk: int):
        if self._prefill_batch_fn is not None:
            return self._prefill_batch_fn(contexts, chunk)
        if self._prefill_single_fn is None:
            raise AttributeError(
                "runner must implement prefill_context or prefill_context_batch"
            )
        results = []
        for ctx in contexts:
            results.append(self._prefill_single_fn(ctx, chunk))
        return results

    def _prefill_overlap_locked(self) -> None:
        if not self._wait_queue or not self._can_accept_ready(0):
            return
        overlap_start = time.perf_counter()
        overlap = self._prefill_until_budget(
            min(self.prefill_chunk, self.max_tokens_per_step),
            skip_ready=False,
        )
        overlap_duration = time.perf_counter() - overlap_start
        consumed = self._last_prefill_tokens
        if overlap:
            self._active.extend(overlap)
        if consumed <= 0:
            return
        metrics = self._last_step_metrics
        metrics["prefill_wall_s"] = (
            metrics.get("prefill_wall_s", 0.0) + overlap_duration
        )
        metrics["prefill_tokens"] = metrics.get("prefill_tokens", 0) + consumed
        metrics["prefill_calls"] = metrics.get("prefill_calls", 0) + 1

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
        for key, value in stats.items():
            if key not in metrics:
                metrics[key] = value
        return metrics

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._wait_queue.clear()
            self._active.clear()
            self._ready.clear()

    @property
    def has_pending_work(self) -> bool:
        with self._lock:
            return (not self._stop) and bool(
                self._wait_queue or self._active or self._ready
            )

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
