# Copyright © 2023-2024 Apple Inc.

import logging
import time
from threading import Lock, Thread
from typing import Callable, Optional

import mlx.core as mx


class ServerStats:
    def __init__(self, log_interval: float = 5.0, pending_counter: Optional[Callable] = None):
        self.log_interval = log_interval
        self._pending_counter = pending_counter
        self._lock = Lock()
        self._running_requests = 0
        self._tokens_generated = 0
        self._prompt_tokens_processed = 0
        self._last_tokens_generated = 0
        self._last_prompt_tokens = 0
        self._last_log_time = time.perf_counter()
        self._stop = False
        self._thread = Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def _log_loop(self):
        while not self._stop:
            time.sleep(self.log_interval)
            if self._stop:
                break
            self._log_stats()

    def _log_stats(self):
        now = time.perf_counter()
        with self._lock:
            elapsed = now - self._last_log_time
            if elapsed <= 0:
                return

            tokens_delta = self._tokens_generated - self._last_tokens_generated
            prompt_delta = self._prompt_tokens_processed - self._last_prompt_tokens
            gen_tps = tokens_delta / elapsed if elapsed > 0 else 0
            prompt_tps = prompt_delta / elapsed if elapsed > 0 else 0
            pending = self._pending_counter() if self._pending_counter else 0

            if self._running_requests > 0 or tokens_delta > 0:
                logging.info(
                    f"Running: {self._running_requests} reqs, "
                    f"Pending: {pending}, "
                    f"Avg generation: {gen_tps:.1f} tok/s, "
                    f"Avg prompt: {prompt_tps:.1f} tok/s"
                )

            self._last_log_time = now
            self._last_tokens_generated = self._tokens_generated
            self._last_prompt_tokens = self._prompt_tokens_processed

    def request_started(self):
        with self._lock:
            self._running_requests += 1

    def request_finished(self):
        with self._lock:
            self._running_requests = max(0, self._running_requests - 1)

    def add_tokens(self, prompt_tokens: int = 0, generation_tokens: int = 0):
        with self._lock:
            self._tokens_generated += generation_tokens
            self._prompt_tokens_processed += prompt_tokens


class RequestMetrics:
    def __init__(self, stats: ServerStats, request_id: str):
        self.stats = stats
        self.request_id = request_id
        self._prompt_start: Optional[float] = None
        self._prompt_end: Optional[float] = None
        self._prompt_tokens = 0
        self._gen_start: Optional[float] = None

    def prompt_progress(self, processed: int, total: int):
        logging.info(f"[{self.request_id}] Prompt processing progress: {processed}/{total}")
        now = time.perf_counter()
        if self._prompt_start is None:
            self._prompt_start = now
        self._prompt_end = now
        self._prompt_tokens = total

    def start_generation(self, prompt_tokens: int):
        now = time.perf_counter()
        if self._prompt_start is None:
            self._prompt_start = now
            self._prompt_tokens = prompt_tokens
        self._gen_start = now
        self.stats.request_started()
        self.stats.add_tokens(prompt_tokens=prompt_tokens)

    def finish(self, generation_tokens: int):
        now = time.perf_counter()
        gen_time = now - self._gen_start if self._gen_start else 0
        gen_tps = generation_tokens / gen_time if gen_time > 0 else 0

        if self._prompt_start and self._prompt_end:
            prompt_time = self._prompt_end - self._prompt_start
            prompt_tps: Optional[float] = (
                self._prompt_tokens / prompt_time if prompt_time > 0 else 0
            )
        else:
            prompt_tps = None

        peak_memory = mx.get_peak_memory() / 1e9
        prompt_tps_str = f"{prompt_tps:.2f}" if prompt_tps is not None else "N/A"
        logging.info(
            f"[{self.request_id}] Finished: "
            f"prompt={self._prompt_tokens} prompt_tps={prompt_tps_str} "
            f"generation={generation_tokens} generation_tps={gen_tps:.2f} "
            f"peak_memory={peak_memory:.3f}GB time={gen_time:.2f}s"
        )
        self.stats.request_finished()
