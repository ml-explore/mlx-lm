# ABOUTME: Coordinates scheduler loop and request streaming for continuous batching.
# ABOUTME: Bridges synchronous HTTP handlers with background worker threads.

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import mlx.core as mx

from ..generate import GenerationResponse
from .engine import ModelRunner
from .scheduler import Scheduler
from .state import SequenceContext

_BENCH_TRACE = os.environ.get("MLXLM_BENCH_TRACE", "").lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}
_BENCH_LOG = logging.getLogger("mlx_lm.bench")
_DEFAULT_KV_BLOCK_SIZE = 16
_DEFAULT_KV_POOL_BLOCKS = 8192
_METAL_PROFILING_ENV = "MLXLM_METAL_PROFILING"
_METAL_PROFILING_ACTIVE = False


def _has_paged_attention_support() -> bool:
    fast = getattr(mx, "fast", None)
    if fast is None:
        return False
    required = ("paged_attention", "_paged_kv_write_impl", "_paged_attention_prewarm")
    return all(hasattr(fast, attr) for attr in required)


def _paged_backend_env_disabled() -> bool:
    value = os.environ.get("MLXLM_PAGED_DISABLE")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _coerce_positive_int(value: Any, name: str, default: int) -> int:
    target = default if value is None else int(value)
    if target <= 0:
        raise ValueError(f"{name} must be positive")
    return target


def _coerce_optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Optional kernel override must be positive when provided")
    return parsed


def _env_truthy(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _should_enable_metal_profiling(config: Dict[str, object]) -> bool:
    env_value = _env_truthy(_METAL_PROFILING_ENV)
    if env_value is not None:
        return env_value
    return bool(config.get("metal_profiling", False))


def _normalize_runtime_config(config: Dict[str, object]) -> None:
    if config is None:
        return

    config["prefill_chunk"] = _coerce_positive_int(
        config.get("prefill_chunk"), "prefill_chunk", 256
    )
    ramp_value = config.get("prefill_ramp_chunk")
    if ramp_value is None:
        config["prefill_ramp_chunk"] = min(config["prefill_chunk"], 64)
    else:
        ramp_int = int(ramp_value)
        if ramp_int <= 0:
            config["prefill_ramp_chunk"] = 0
        else:
            config["prefill_ramp_chunk"] = min(ramp_int, config["prefill_chunk"])

    hybrid_value = config.get("prefill_hybrid_threshold", 0)
    try:
        hybrid_int = int(hybrid_value)
    except (TypeError, ValueError):
        hybrid_int = 0
    config["prefill_hybrid_threshold"] = max(
        0, min(hybrid_int, config["prefill_chunk"])
    )

    budget_value = config.get("prefill_ramp_budget_ms", None)
    if budget_value is None:
        config["prefill_ramp_budget_ms"] = None
    else:
        try:
            budget_float = float(budget_value)
        except (TypeError, ValueError):
            budget_float = None
        if budget_float is not None and budget_float > 0:
            config["prefill_ramp_budget_ms"] = budget_float
        else:
            config["prefill_ramp_budget_ms"] = None

    attn_backend = str(config.get("attn_backend", "auto")).lower()
    if attn_backend not in {"auto", "dense", "paged"}:
        raise ValueError("attn_backend must be one of {'auto', 'dense', 'paged'}")

    config["kv_block_size"] = _coerce_positive_int(
        config.get("kv_block_size"), "kv_block_size", _DEFAULT_KV_BLOCK_SIZE
    )
    kv_pool_value = config.get("kv_pool_blocks", "auto")
    config["kv_pool_blocks"] = _coerce_optional_positive_int(kv_pool_value)
    config["paged_vec_width"] = _coerce_optional_positive_int(
        config.get("paged_vec_width")
    )
    config["paged_threads_per_head"] = _coerce_optional_positive_int(
        config.get("paged_threads_per_head")
    )
    kv_quant_mode = str(config.get("kv_quant_mode", "none")).lower()
    if kv_quant_mode not in {"none", "int4_v"}:
        raise ValueError("kv_quant_mode must be one of {'none', 'int4_v'}")
    config["kv_quant_mode"] = kv_quant_mode
    config["kv_quant_group_size"] = _coerce_positive_int(
        config.get("kv_quant_group_size"), "kv_quant_group_size", 64
    )
    config["decode_unroll"] = _coerce_positive_int(
        config.get("decode_unroll"), "decode_unroll", 1
    )
    config["decode_unroll_safe"] = bool(config.get("decode_unroll_safe", True))

    paged_supported = _has_paged_attention_support()
    if attn_backend == "paged" and not paged_supported:
        raise RuntimeError(
            "Paged attention backend requested but not supported by current MLX build"
        )
    if attn_backend == "auto":
        selected = "paged" if paged_supported else "dense"
    else:
        selected = attn_backend

    env_disabled = _paged_backend_env_disabled()
    if env_disabled and selected == "paged":
        logging.warning("MLXLM_PAGED_DISABLE set; forcing dense attention backend")
        selected = "dense"

    config["attn_backend"] = attn_backend
    config["selected_attn_backend"] = selected
    config["paged_backend_available"] = paged_supported
    config["paged_backend_enabled"] = selected == "paged"
    config["paged_backend_env_disabled"] = env_disabled
    config["metal_profiling"] = _should_enable_metal_profiling(config)


def _maybe_enable_metal_profiling(requested: bool) -> None:
    global _METAL_PROFILING_ACTIVE
    if not requested or _METAL_PROFILING_ACTIVE:
        return
    metal = getattr(mx, "metal", None)
    if metal is None:
        return
    if not getattr(metal, "command_buffer_profiling_supported", lambda: False)():
        return
    try:
        metal.set_command_buffer_profiling(True)
        _METAL_PROFILING_ACTIVE = True
    except Exception:  # pragma: no cover - defensive
        logging.debug("Unable to enable Metal profiling", exc_info=True)


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
            prefill_queue_limit=getattr(runner, "prefill_buffer", 0),
            prefill_slice_cap=getattr(runner, "prefill_slice_cap", prefill_chunk),
        )
        self._debug_metrics = debug_metrics
        self._metrics_history: list[dict] = []
        self._metrics_lock = threading.Lock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._worker_exception: Optional[BaseException] = None
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
            try:
                if self.scheduler.has_pending_work:
                    self.scheduler.step()
                    metrics = self.scheduler.metrics
                    self._record_metrics(metrics)
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
                        if _BENCH_TRACE:
                            kv_blocks = metrics.get("kv_pool_blocks")
                            kv_auto = metrics.get("kv_pool_auto_blocks")
                            _BENCH_LOG.info(
                                "tick.detail decode_batch=%s decode_tokens=%s prefill_tokens=%s kv_blocks=%s kv_auto=%s",
                                metrics.get("decode_batch_size", 0),
                                metrics.get("decode_tokens", 0),
                                metrics.get("prefill_tokens", 0),
                                kv_blocks,
                                kv_auto,
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
            except BaseException as exc:  # pragma: no cover - defensive guard
                self._worker_exception = exc
                logging.exception("continuous batching worker failed", exc_info=True)
                self._shutdown.set()
                self._wake.set()
                break

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

    def _record_metrics(self, metrics: dict) -> None:
        if not metrics:
            return
        snapshot = dict(metrics)
        snapshot.setdefault("timestamp", time.perf_counter())
        with self._metrics_lock:
            self._metrics_history.append(snapshot)

    def metrics_history(self) -> list[dict]:
        with self._metrics_lock:
            return [dict(entry) for entry in self._metrics_history]

    def _stream(self, ctx: SequenceContext):
        while True:
            if self._worker_exception is not None:
                raise RuntimeError(
                    "continuous batching worker failed; check logs for details"
                ) from self._worker_exception
            if ctx.state.cancel_requested and ctx.events.empty():
                yield self._cancel_event(ctx)
                break

            try:
                event = ctx.events.get(timeout=0.1)
            except queue.Empty:
                continue
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
        vocab_size = getattr(self.runner.tokenizer, "vocab_size", 1) or 1
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
    config: Dict[str, object],
    *,
    model,
    tokenizer,
    draft_model=None,
) -> Optional[ContinuousBatchingRuntime]:
    if config is None:
        return None
    _normalize_runtime_config(config)
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
        prefill_ramp_chunk=config.get("prefill_ramp_chunk"),
        prefill_hybrid_threshold=config.get("prefill_hybrid_threshold", 0),
        prefill_ramp_budget_ms=config.get("prefill_ramp_budget_ms"),
        force_legacy_generator=bool(config.get("force_legacy_generator", False)),
        attn_backend=str(config.get("selected_attn_backend", "dense")),
        kv_block_size=int(config.get("kv_block_size", 16)),
        kv_pool_blocks=config.get("kv_pool_blocks"),
        paged_vec_width=config.get("paged_vec_width"),
        paged_threads_per_head=config.get("paged_threads_per_head"),
        kv_quant_mode=str(config.get("kv_quant_mode", "none")),
        kv_quant_group_size=int(config.get("kv_quant_group_size", 64)),
        decode_unroll=int(config.get("decode_unroll", 1)),
        decode_engine=str(config.get("decode_engine", "dense")),
    )
    _maybe_enable_metal_profiling(bool(config.get("metal_profiling")))
    return ContinuousBatchingRuntime(
        runner=runner,
        max_num_seqs=config["max_num_seqs"],
        max_tokens_per_step=config["max_tokens_per_step"],
        prefill_chunk=config["prefill_chunk"],
    )


__all__ = ["ContinuousBatchingRuntime", "create_runtime"]
