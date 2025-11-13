# ABOUTME: Benchmarks static batch_generate against continuous batching runtime.
# ABOUTME: Uses wall-clock throughput under Poisson arrivals to track gains.

"""
Benchmark static batch_generate vs continuous batching runtime under Poisson arrivals.
Run: python bench/bench_continuous_vs_static.py --repo mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx

os.environ.setdefault("MLX_METAL_PROFILING", "1")

if hasattr(mx, "metal") and hasattr(mx.metal, "set_command_buffer_profiling"):
    try:
        if getattr(mx.metal, "command_buffer_profiling_supported", lambda: False)():
            if not mx.metal.command_buffer_profiling_enabled():
                mx.metal.set_command_buffer_profiling(True)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        logging.warning("Unable to enable Metal profiling for bench: %s", exc)

random.seed(0)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# TODO: Drop this fallback once paged attention lands and dense slabs disappear.
KV_SLAB_LIMIT_BYTES = int(os.environ.get("MLX_KV_SLAB_LIMIT_MB", "320")) * 1024 * 1024
BENCH_TRACE = os.environ.get("MLXLM_BENCH_TRACE", "").lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}
BENCH_LOG = logging.getLogger("mlx_lm.bench")
EMIT_TRACE = os.environ.get("MLXLM_BENCH_EMIT_TRACE", "").lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}

try:
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.server_batched.engine import ModelRunner
    from mlx_lm.server_batched.graph_decode.llama_prefill import LlamaPrefillGraph
    from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime
except Exception as exc:  # pragma: no cover - exercised when deps missing during tests
    BENCH_LOG.debug("mlx_lm imports unavailable: %s", exc)
    batch_generate = None
    load = None
    make_sampler = None
    LlamaPrefillGraph = None
    ModelRunner = None
    ContinuousBatchingRuntime = None


@dataclass
class Result:
    submit_ns: int
    first_token_ns: Optional[int]
    finish_ns: int
    gen_tokens: int
    emit_trace: List[Dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class TokenControls:
    sampler_settings: Dict[str, float]
    static_stop_tokens: List[int]
    runtime_stop_tokens: Set[int]
    use_eos_stop: bool


@contextmanager
def _temporary_env(var: str, value: str):
    original = os.environ.get(var)
    try:
        os.environ[var] = value
        yield
    finally:
        if original is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = original


def _base_sampler_settings(stop_tokens: List[int]) -> Dict[str, float]:
    return {
        "temp": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 0,
        "xtc_probability": 0.0,
        "xtc_threshold": 0.0,
        "xtc_special_tokens": list(stop_tokens),
    }


def select_token_controls(
    *,
    mode: str,
    tokenizer,
    explicit_stops: Optional[Set[int]],
) -> TokenControls:
    if mode not in {"fixed", "eos"}:
        raise ValueError(f"Unsupported mode {mode}")

    if mode == "fixed":
        sampler_settings = _base_sampler_settings([])
        return TokenControls(
            sampler_settings=sampler_settings,
            static_stop_tokens=[],
            runtime_stop_tokens=set(),
            use_eos_stop=False,
        )

    if explicit_stops:
        stop_tokens = sorted(explicit_stops)
    else:
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids:
            stop_tokens = sorted(int(tok) for tok in eos_ids)
        else:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("Tokenizer must expose eos_token_id for EOS mode")
            stop_tokens = [int(eos_id)]

    sampler_settings = _base_sampler_settings(stop_tokens)
    return TokenControls(
        sampler_settings=sampler_settings,
        static_stop_tokens=list(stop_tokens),
        runtime_stop_tokens=set(stop_tokens),
        use_eos_stop=True,
    )


def normalise_request_count(n_value: Optional[int], concurrency: int) -> int:
    if n_value is None:
        return concurrency
    return n_value


def compute_steady_state_tps(
    history: List[Dict[str, float]],
    *,
    steady_fraction: float,
    max_num_seqs: int,
) -> float:
    if not history or steady_fraction <= 0 or max_num_seqs <= 0:
        return 0.0
    threshold = steady_fraction * max_num_seqs
    samples: List[float] = []
    for entry in history:
        active = entry.get("active_sequences", 0)
        duration = entry.get("decode_duration_s", 0.0)
        tokens = entry.get("decode_tokens", 0)
        if active < threshold:
            continue
        if duration <= 0 or tokens <= 0:
            continue
        samples.append(tokens / duration)
    if not samples:
        return 0.0
    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2:
        return samples[mid]
    return (samples[mid - 1] + samples[mid]) / 2.0


def phase_summaries(
    history: List[Dict[str, float]],
    *,
    steady_fraction: float,
    max_num_seqs: int,
    spike_threshold_s: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    if not history or steady_fraction <= 0 or max_num_seqs <= 0:
        return {}

    threshold = steady_fraction * max_num_seqs
    first_steady = None
    last_steady = None
    for idx, tick in enumerate(history):
        if tick.get("active_sequences", 0) >= threshold:
            if first_steady is None:
                first_steady = idx
            last_steady = idx

    if first_steady is None:
        ramp_indices = list(range(len(history)))
        steady_indices: List[int] = []
        tail_indices: List[int] = []
    else:
        ramp_indices = list(range(0, first_steady))
        steady_indices = list(range(first_steady, last_steady + 1))
        tail_indices = list(range(last_steady + 1, len(history)))

    def aggregate(indices: List[int]) -> Dict[str, float]:
        ticks = len(indices)
        if ticks == 0:
            return {
                "ticks": 0,
                "time_s": 0.0,
                "tokens": 0,
                "tokens_per_sec": 0.0,
                "mean_decode_ms": 0.0,
                "mean_batch": 0.0,
                "prefill_tokens": 0,
                "prefill_wall_s": 0.0,
                "decode_model_ms": 0.0,
                "decode_driver_ms": 0.0,
                "array_prefill_graph_s": 0.0,
                "array_prefill_writer_s": 0.0,
                "array_prefill_chunk_ms_total": 0.0,
                "array_prefill_chunk_count": 0.0,
                "array_prefill_chunk_ms_mean": 0.0,
                "array_prefill_first_chunk_ms": 0.0,
                "paged_prefill_slice_s": 0.0,
                "paged_prefill_model_s": 0.0,
                "paged_prefill_commit_s": 0.0,
                "paged_prefill_slice_count": 0.0,
                "paged_prefill_first_slice_ms": 0.0,
            }
        total_time = sum(history[i].get("decode_duration_s", 0.0) for i in indices)
        total_tokens = sum(history[i].get("decode_tokens", 0) for i in indices)
        mean_batch = (
            sum(
                history[i].get(
                    "decode_batch_size", history[i].get("active_sequences", 0)
                )
                for i in indices
            )
            / ticks
        )
        total_prefill = sum(history[i].get("prefill_tokens", 0) for i in indices)
        total_prefill_wall = sum(history[i].get("prefill_wall_s", 0.0) for i in indices)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        mean_decode_ms = (total_time / ticks) * 1000.0 if ticks else 0.0
        total_model = sum(
            history[i].get("decode_model_duration_s", 0.0) for i in indices
        )
        total_driver = sum(
            history[i].get("decode_total_duration_s", 0.0) for i in indices
        )
        total_array_graph = sum(
            history[i].get("array_prefill_graph_s", 0.0) for i in indices
        )
        total_array_writer = sum(
            history[i].get("array_prefill_writer_s", 0.0) for i in indices
        )
        max_pending_tokens = 0.0
        last_pending_tokens = 0.0
        if ticks:
            max_pending_tokens = max(
                history[i].get("array_prefill_pending_tokens_max", 0.0) for i in indices
            )
            last_pending_tokens = history[indices[-1]].get(
                "array_prefill_pending_tokens", 0.0
            )
        total_chunk_ms = sum(
            history[i].get("array_prefill_chunk_ms_total", 0.0) for i in indices
        )
        total_chunk_count = sum(
            history[i].get("array_prefill_chunk_count", 0.0) for i in indices
        )
        total_attn = sum(history[i].get("array_prefill_attn_s", 0.0) for i in indices)
        total_mlp = sum(history[i].get("array_prefill_mlp_s", 0.0) for i in indices)
        total_overlay = sum(
            history[i].get("array_prefill_overlay_s", 0.0) for i in indices
        )
        total_overlay_wait = sum(
            history[i].get("array_prefill_overlay_wait_s", 0.0) for i in indices
        )
        total_overlay_wait_count = sum(
            history[i].get("array_prefill_overlay_wait_count", 0.0) for i in indices
        )
        waiting_values = [
            history[i].get("array_prefill_waiting_sequences", 0.0) for i in indices
        ]
        waiting_max = max(waiting_values) if waiting_values else 0.0
        waiting_last = waiting_values[-1] if waiting_values else 0.0
        total_paged_slice = sum(
            history[i].get("paged_prefill_slice_s", 0.0) for i in indices
        )
        total_paged_model = sum(
            history[i].get("paged_prefill_model_s", 0.0) for i in indices
        )
        total_paged_commit = sum(
            history[i].get("paged_prefill_commit_s", 0.0) for i in indices
        )
        total_paged_slice_count = sum(
            history[i].get("paged_prefill_slice_count", 0.0) for i in indices
        )
        first_chunk_ms = 0.0
        for idx in indices:
            value = history[idx].get("array_prefill_first_chunk_ms", 0.0)
            if value:
                first_chunk_ms = value
                break
        first_paged_slice_ms = 0.0
        for idx in indices:
            value = history[idx].get("paged_prefill_first_slice_ms", 0.0)
            if value:
                first_paged_slice_ms = value
                break
        return {
            "ticks": ticks,
            "time_s": total_time,
            "tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "mean_decode_ms": mean_decode_ms,
            "mean_batch": mean_batch,
            "prefill_tokens": total_prefill,
            "prefill_wall_s": total_prefill_wall,
            "decode_model_ms": (total_model / ticks) * 1000.0 if ticks else 0.0,
            "decode_driver_ms": (total_driver / ticks) * 1000.0 if ticks else 0.0,
            "array_prefill_graph_s": total_array_graph,
            "array_prefill_writer_s": total_array_writer,
            "array_prefill_chunk_ms_total": total_chunk_ms,
            "array_prefill_chunk_count": total_chunk_count,
            "array_prefill_chunk_ms_mean": (
                total_chunk_ms / total_chunk_count if total_chunk_count else 0.0
            ),
            "array_prefill_first_chunk_ms": first_chunk_ms,
            "array_prefill_pending_tokens_max": max_pending_tokens,
            "array_prefill_pending_tokens_last": last_pending_tokens,
            "array_prefill_attn_s": total_attn,
            "array_prefill_mlp_s": total_mlp,
            "array_prefill_overlay_s": total_overlay,
            "array_prefill_overlay_wait_s": total_overlay_wait,
            "array_prefill_overlay_wait_count": total_overlay_wait_count,
            "array_prefill_waiting_sequences_max": waiting_max,
            "array_prefill_waiting_sequences_last": waiting_last,
            "paged_prefill_slice_s": total_paged_slice,
            "paged_prefill_model_s": total_paged_model,
            "paged_prefill_commit_s": total_paged_commit,
            "paged_prefill_slice_count": total_paged_slice_count,
            "paged_prefill_first_slice_ms": first_paged_slice_ms,
        }

    summaries = {
        "ramp": aggregate(ramp_indices),
        "steady": aggregate(steady_indices),
        "tail": aggregate(tail_indices),
    }

    spike_ticks = [
        tick
        for tick in history
        if tick.get("decode_duration_s", 0.0) > spike_threshold_s
    ]
    spike_time = sum(tick.get("decode_duration_s", 0.0) for tick in spike_ticks)
    summaries["spikes"] = {
        "count": len(spike_ticks),
        "time_s": spike_time,
        "threshold_s": spike_threshold_s,
    }

    return summaries


def _model_layers(model):
    core = getattr(model, "model", model)
    return getattr(core, "layers", None)


def collect_prefill_profile_metrics(
    model, chunk_len: int, ramp_chunk: int
) -> Optional[Dict[str, object]]:
    chunk = max(1, int(chunk_len))
    ramp = max(1, min(chunk, int(ramp_chunk if ramp_chunk else chunk)))
    graph_cls = LlamaPrefillGraph
    if graph_cls is None:
        BENCH_LOG.warning("Prefill profile unavailable: LlamaPrefillGraph missing")
        return None
    try:
        graph = graph_cls(model)
    except Exception as exc:  # pragma: no cover - exercised via error-path test
        BENCH_LOG.warning("Prefill profile unavailable: %s", exc)
        return None

    layers = _model_layers(model)
    if not layers:
        BENCH_LOG.warning("Prefill profile skipped: model has no layers attribute")
        return None
    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        BENCH_LOG.warning("Prefill profile skipped: first layer missing self_attn")
        return None
    n_layers = len(layers)
    n_kv_heads = getattr(attn, "n_kv_heads", None)
    if not n_kv_heads:
        BENCH_LOG.warning("Prefill profile skipped: n_kv_heads undefined")
        return None
    dtype_attr = getattr(getattr(attn, "q_proj", None), "weight", None)
    cache_dtype = getattr(dtype_attr, "dtype", mx.float16)
    batch = 1
    block_tables_shape = (batch, 1)
    k_cache_shape = (n_layers, n_kv_heads, batch, chunk, graph.head_dim)
    overlay_shape = (n_layers, chunk, batch, n_kv_heads, graph.head_dim)
    try:
        with _temporary_env("MLXLM_PREFILL_PROFILE", "1"):
            fn = graph.get_compiled(
                batch_size=batch,
                chunk_len=chunk,
                block_tables_shape=block_tables_shape,
                k_cache_shape=k_cache_shape,
                v_cache_shape=k_cache_shape,
                kv_map_shape=None,
                dtype=mx.float16,
                pending_flag=0,
            )
            tokens = mx.zeros((batch, chunk), dtype=mx.int32)
            base_lens = mx.zeros((batch,), dtype=mx.int32)
            block_tables = mx.zeros(block_tables_shape, dtype=mx.int32)
            k_cache = mx.zeros(k_cache_shape, dtype=cache_dtype)
            v_cache = mx.zeros_like(k_cache)
            pending_k = mx.zeros(overlay_shape, dtype=cache_dtype)
            pending_v = mx.zeros_like(pending_k)
            fn(tokens, base_lens, block_tables, k_cache, v_cache, pending_k, pending_v)
    except Exception as exc:  # pragma: no cover - exercised via error-path test
        BENCH_LOG.warning("Prefill profile execution failed: %s", exc)
        return None
    metrics = graph.consume_metrics()
    return {
        "chunk_len": chunk,
        "prefill_ramp_chunk": ramp,
        "array_prefill_attn_s": metrics.get("array_prefill_attn_s", 0.0),
        "array_prefill_mlp_s": metrics.get("array_prefill_mlp_s", 0.0),
        "array_prefill_overlay_s": metrics.get("array_prefill_overlay_s", 0.0),
        "array_prefill_layer_attn_s": metrics.get("array_prefill_layer_attn_s", []),
        "array_prefill_layer_mlp_s": metrics.get("array_prefill_layer_mlp_s", []),
        "array_prefill_layer_overlay_s": metrics.get(
            "array_prefill_layer_overlay_s", []
        ),
    }


def extract_apc_stats(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    keys = (
        "prefix_hits",
        "prefix_lookups",
        "prefix_hit_rate",
        "prefix_tokens_reused",
    )
    for entry in reversed(history):
        stats = {key: entry.get(key) for key in keys if key in entry}
        if stats:
            return stats
    return {}


def extract_compile_stats(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    keys = (
        "compile_cache_hits",
        "compile_cache_misses",
        "array_phase1_compile_hits",
        "array_phase1_compile_misses",
        "array_phase2_compile_hits",
        "array_phase2_compile_misses",
        "array_phase1_duration_s",
        "array_phase2_attention_duration_s",
        "array_phase2_outproj_duration_s",
        "array_phase2_mlp_duration_s",
        "array_writer_duration_s",
        "array_phase2_compiled_duration_s",
    )
    for entry in reversed(history):
        stats = {key: entry.get(key) for key in keys if key in entry}
        if stats:
            return stats
    return {}


def should_schedule_open_loop(
    backlog: int,
    *,
    target_tokens: Optional[int],
    total_tokens_emitted: int,
    backlog_limit: Optional[int] = None,
) -> bool:
    if backlog_limit is not None and backlog >= backlog_limit:
        if BENCH_TRACE:
            BENCH_LOG.info(
                "open_loop.skip reason=backlog backlog=%s limit=%s active=%s free=%s",
                backlog,
                backlog_limit,
                "n/a",
                "n/a",
            )
        return False
    if target_tokens is not None and total_tokens_emitted >= target_tokens:
        if BENCH_TRACE:
            BENCH_LOG.info(
                "open_loop.skip reason=target_reached emitted=%s target=%s",
                total_tokens_emitted,
                target_tokens,
            )
        return False
    return True


def _extract_layers(model) -> Optional[List]:
    if hasattr(model, "layers") and model.layers:
        return list(model.layers)
    core = getattr(model, "model", None)
    if core is not None and hasattr(core, "layers") and core.layers:
        return list(core.layers)
    return None


def estimate_safe_max_num_seqs(
    model,
    requested: int,
    max_tokens: int,
    *,
    limit_bytes: int = KV_SLAB_LIMIT_BYTES,
) -> Tuple[int, Optional[Dict[str, int]]]:
    if requested <= 0 or max_tokens <= 0 or limit_bytes <= 0:
        return max(1, requested), None

    layers = _extract_layers(model)
    if not layers:
        return requested, None

    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        return requested, None

    n_kv_heads = getattr(attn, "n_kv_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if not n_kv_heads or not head_dim:
        return requested, None

    dtype = None
    if hasattr(attn, "k_proj") and hasattr(attn.k_proj, "weight"):
        dtype = attn.k_proj.weight.dtype
    elif hasattr(attn, "v_proj") and hasattr(attn.v_proj, "weight"):
        dtype = attn.v_proj.weight.dtype
    dtype_size = getattr(dtype, "size", 2) if dtype is not None else 2

    per_token_bytes = n_kv_heads * head_dim * dtype_size * 2
    per_slot_bytes = per_token_bytes * max_tokens * len(layers)
    if per_slot_bytes <= 0:
        return requested, None

    safe_slots = int(limit_bytes // per_slot_bytes)
    safe_slots = max(1, safe_slots)
    if safe_slots >= requested:
        return requested, None

    meta = {
        "limit_bytes": int(limit_bytes),
        "per_slot_bytes": int(per_slot_bytes),
        "dtype_size": int(dtype_size),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(head_dim),
        "num_layers": int(len(layers)),
    }
    return safe_slots, meta


def poisson_arrivals(lmbda: float, n: int):
    t = 0.0
    out = []
    for _ in range(n):
        u = random.random()
        gap = -math.log1p(-u) / lmbda
        t += gap
        out.append(t)
    return out


def closed_arrival_schedule(count: int, mode: str, lmbda: float) -> List[float]:
    if count <= 0:
        return []
    if mode == "burst":
        return [0.0] * count
    if mode == "poisson":
        return poisson_arrivals(lmbda, count)
    raise ValueError(f"Unsupported arrival mode {mode!r}")


_PAGED_DECODE_ENGINES = {"paged", "paged-arrays", "paged-arrays+compile"}


def should_use_runtime_static(decode_engine: str, static_mode: str) -> bool:
    if static_mode == "runtime":
        return True
    if static_mode == "batch":
        return False
    return decode_engine in _PAGED_DECODE_ENGINES


def summarize(
    name: str, results: List[Result], *, ttft_window: Optional[int] = None
) -> Dict[str, float]:
    if not results:
        print(
            f"[{name}] n=0 tokens=0 wall=0.00s tokens/s=0.00 ttft mean=0.0ms median=0.0ms p95=0.0ms"
        )
        return {
            "count": 0,
            "tokens": 0,
            "wall_seconds": 0.0,
            "tokens_per_sec": 0.0,
            "ttft_mean_ms": 0.0,
            "ttft_median_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "mean_tokens_per_request": 0.0,
            "ttft_samples_ms": [],
            "ttft_window_count": 0,
            "ttft_window_mean_ms": 0.0,
            "ttft_window_median_ms": 0.0,
            "ttft_window_p95_ms": 0.0,
        }

    start_ns = min(r.submit_ns for r in results)
    end_ns = max(r.finish_ns for r in results)
    wall_seconds = max((end_ns - start_ns) / 1e9, 1e-9)
    total_tokens = sum(r.gen_tokens for r in results)
    tokens_per_sec = total_tokens / wall_seconds

    ttft_pairs = []
    for r in results:
        first = r.first_token_ns if r.first_token_ns is not None else r.finish_ns
        ttft_ms = max((first - r.submit_ns) / 1e6, 0.0)
        ttft_pairs.append((r.finish_ns, ttft_ms))

    def _p95(values: List[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = max(int(math.ceil(0.95 * len(ordered))) - 1, 0)
        return ordered[idx]

    ordered_by_finish = [val for _, val in sorted(ttft_pairs, key=lambda pair: pair[0])]
    ordered_all = sorted(ordered_by_finish)
    ttft_mean = mean(ordered_all) if ordered_all else 0.0
    ttft_median = median(ordered_all) if ordered_all else 0.0
    ttft_p95 = _p95(ordered_all)

    limit = (
        int(ttft_window)
        if ttft_window is not None and ttft_window > 0
        else len(ordered_by_finish)
    )
    window_samples = ordered_by_finish[:limit]
    window_mean = mean(window_samples) if window_samples else 0.0
    window_median = median(window_samples) if window_samples else 0.0
    window_p95 = _p95(window_samples)

    mean_tokens = total_tokens / len(results) if results else 0.0
    ttft_display = ", ".join(f"{val:.1f}" for val in ordered_all)
    print(
        f"[{name}] n={len(results)} tokens={total_tokens} wall={wall_seconds:.2f}s "
        f"tokens/s={tokens_per_sec:.2f} "
        f"ttft mean={ttft_mean:.1f}ms median={ttft_median:.1f}ms p95={ttft_p95:.1f}ms "
        f"window_median={window_median:.1f}ms window_count={len(window_samples)} "
        f"gen/req mean={mean_tokens:.1f}"
    )
    print(f"[{name}] ttft_samples_ms=[{ttft_display}]")
    return {
        "count": len(results),
        "tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_sec": tokens_per_sec,
        "ttft_mean_ms": ttft_mean,
        "ttft_median_ms": ttft_median,
        "ttft_p95_ms": ttft_p95,
        "mean_tokens_per_request": mean_tokens,
        "ttft_samples_ms": ordered_all,
        "ttft_window_count": len(window_samples),
        "ttft_window_mean_ms": window_mean,
        "ttft_window_median_ms": window_median,
        "ttft_window_p95_ms": window_p95,
    }


def emit_trace_summary(
    results: List[Result], sample_limit: int = 8
) -> Dict[str, object]:
    if not results:
        return {
            "emit_event_counts": [],
            "emit_trace_samples": [],
        }
    counts = [len(r.emit_trace) for r in results]
    samples = [
        list(r.emit_trace)
        for r in results[: min(sample_limit, len(results))]
        if r.emit_trace
    ]
    return {
        "emit_event_counts": counts,
        "emit_trace_samples": samples,
    }


def _maybe_kill_switch(
    *,
    args,
    static_summary: Dict[str, float],
    continuous_summary: Dict[str, float],
    apc_stats: Optional[Dict[str, float]],
) -> List[str]:
    if args is None:
        return []
    reasons: List[str] = []
    static_tps = static_summary.get("tokens_per_sec", 0.0)
    cont_tps = continuous_summary.get("tokens_per_sec", 0.0)
    ratio = (cont_tps / static_tps) if static_tps > 0 else float("inf")
    if (
        args.min_throughput_ratio > 0
        and args.concurrency >= 4
        and static_tps > 0
        and ratio < args.min_throughput_ratio
    ):
        reasons.append(
            f"continuous tokens/s {cont_tps:.2f} < "
            f"{args.min_throughput_ratio:.2f}x static {static_tps:.2f}"
        )
    if (
        args.min_apc_hit_rate > 0
        and apc_stats
        and apc_stats.get("prefix_lookups", 0.0) > 0
        and apc_stats.get("prefix_hit_rate", 0.0) < args.min_apc_hit_rate
    ):
        reasons.append(
            "APC hit-rate {hit:.2%} < required {req:.2%}".format(
                hit=apc_stats["prefix_hit_rate"],
                req=args.min_apc_hit_rate,
            )
        )
    return reasons


def run_static(
    model,
    tokenizer,
    prompts,
    max_tokens,
    *,
    stop_tokens,
    sampler_settings,
    open_loop: bool,
    tokens_target: Optional[int],
):
    if batch_generate is None or make_sampler is None:
        raise RuntimeError(
            "mlx_lm batch APIs unavailable; install mlx-lm to run the benchmark"
        )
    prompt_token_batches = [tokenizer.encode(p) for p in prompts]
    results: List[Result] = []
    total_tokens = 0
    target_tokens = (
        tokens_target if (open_loop and tokens_target and tokens_target > 0) else None
    )
    sampler = make_sampler(**sampler_settings)
    with _override_tokenizer_stops(tokenizer, stop_tokens):
        batch_iter = itertools.cycle(prompt_token_batches)
        while True:
            start_ns = time.perf_counter_ns()
            response = batch_generate(
                model,
                tokenizer,
                [next(batch_iter) for _ in prompt_token_batches],
                max_tokens=max_tokens,
                verbose=False,
                sampler=sampler,
            )
            end_ns = time.perf_counter_ns()
            batch_tokens = 0
            for text in response.texts:
                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                batch_tokens += tokens
                results.append(
                    Result(
                        submit_ns=start_ns,
                        first_token_ns=end_ns,
                        finish_ns=end_ns,
                        gen_tokens=tokens,
                        emit_trace=[
                            {
                                "tokens": tokens,
                                "token_id": None,
                                "finish_reason": "length",
                            }
                        ],
                    )
                )
            total_tokens += batch_tokens
            if not open_loop:
                break
            if target_tokens is not None and total_tokens >= target_tokens:
                break
    return results


def run_static_runtime(
    model,
    tokenizer,
    prompts,
    max_tokens,
    lmbda,
    *,
    max_num_seqs: int,
    prefill_chunk: int,
    prefill_ramp_chunk: int,
    max_tokens_per_step: int,
    decode_unroll: int,
    stop_tokens_override: Optional[set],
    use_eos_stop: bool,
    sampler_settings: Dict[str, float],
    attn_backend: str,
    kv_block_size: int,
    kv_pool_blocks: Optional[int],
    paged_vec_width: Optional[int],
    paged_threads_per_head: Optional[int],
    kv_quant_mode: str,
    kv_quant_group_size: int,
    decode_engine: str,
    prefill_hybrid_threshold: int,
    prefill_ramp_budget_ms: Optional[float],
):
    """Runs the runtime in closed-loop mode to measure paged static TTFT."""

    results, _ = run_continuous(
        model,
        tokenizer,
        prompts,
        max_tokens,
        lmbda,
        max_num_seqs=max_num_seqs,
        prefill_chunk=prefill_chunk,
        prefill_ramp_chunk=prefill_ramp_chunk,
        max_tokens_per_step=max_tokens_per_step,
        decode_unroll=decode_unroll,
        stop_tokens_override=stop_tokens_override,
        use_eos_stop=use_eos_stop,
        sampler_settings=sampler_settings,
        open_loop=False,
        tokens_target=None,
        attn_backend=attn_backend,
        kv_block_size=kv_block_size,
        kv_pool_blocks=kv_pool_blocks,
        paged_vec_width=paged_vec_width,
        paged_threads_per_head=paged_threads_per_head,
        kv_quant_mode=kv_quant_mode,
        kv_quant_group_size=kv_quant_group_size,
        open_loop_backlog_limit=None,
        decode_engine=decode_engine,
        prefill_hybrid_threshold=prefill_hybrid_threshold,
        prefill_ramp_budget_ms=prefill_ramp_budget_ms,
        arrival_mode="burst",
    )
    return results


@contextmanager
def _override_tokenizer_stops(tokenizer, stop_tokens: List[int]):
    original_ids = getattr(tokenizer, "eos_token_ids", None)
    had_ids = hasattr(tokenizer, "eos_token_ids")
    original_id = getattr(tokenizer, "eos_token_id", None)
    had_id = hasattr(tokenizer, "eos_token_id")
    try:
        override_ids = list(stop_tokens)
        override_id = override_ids[0] if override_ids else None
        setattr(tokenizer, "eos_token_ids", override_ids)
        setattr(tokenizer, "eos_token_id", override_id)
        yield
    finally:
        if had_ids:
            setattr(tokenizer, "eos_token_ids", original_ids)
        else:
            try:
                delattr(tokenizer, "eos_token_ids")
            except AttributeError:
                pass
        if had_id:
            setattr(tokenizer, "eos_token_id", original_id)
        else:
            try:
                delattr(tokenizer, "eos_token_id")
            except AttributeError:
                pass


def run_continuous(
    model,
    tokenizer,
    prompts,
    max_tokens,
    lmbda,
    *,
    max_num_seqs: int,
    prefill_chunk: int,
    prefill_ramp_chunk: int,
    max_tokens_per_step: int,
    decode_unroll: int,
    stop_tokens_override: Optional[set],
    use_eos_stop: bool,
    sampler_settings: Dict[str, float],
    open_loop: bool,
    tokens_target: Optional[int],
    attn_backend: str,
    kv_block_size: int,
    kv_pool_blocks: Optional[int],
    paged_vec_width: Optional[int],
    paged_threads_per_head: Optional[int],
    kv_quant_mode: str,
    kv_quant_group_size: int,
    open_loop_backlog_limit: Optional[int] = None,
    decode_engine: str = "dense",
    prefill_hybrid_threshold: int = 0,
    prefill_ramp_budget_ms: Optional[float] = None,
    arrival_mode: str = "burst",
):
    if ModelRunner is None or ContinuousBatchingRuntime is None:
        raise RuntimeError(
            "mlx_lm continuous runtime unavailable; install mlx-lm to run the benchmark"
        )
    # Force paged backend when array engines are requested so the runtime
    # actually builds the paged manager/arrays.
    effective_attn_backend = attn_backend
    if decode_engine in {"paged", "paged-arrays", "paged-arrays+compile"}:
        effective_attn_backend = "paged"
    runner = ModelRunner(
        model,
        tokenizer,
        draft_model=None,
        max_num_seqs=max_num_seqs,
        prefill_chunk=prefill_chunk,
        stop_tokens=stop_tokens_override,
        attn_backend=effective_attn_backend,
        kv_block_size=kv_block_size,
        kv_pool_blocks=kv_pool_blocks,
        paged_vec_width=paged_vec_width,
        paged_threads_per_head=paged_threads_per_head,
        kv_quant_mode=kv_quant_mode,
        kv_quant_group_size=kv_quant_group_size,
        decode_unroll=decode_unroll,
        prefill_ramp_chunk=prefill_ramp_chunk,
        prefill_hybrid_threshold=prefill_hybrid_threshold,
        prefill_ramp_budget_ms=prefill_ramp_budget_ms,
        decode_engine=decode_engine,
    )
    runner.open_loop_mode = open_loop
    runner.open_loop_draining = False
    logging.info("runtime stop_tokens=%s", sorted(runner.stop_tokens))
    runtime = ContinuousBatchingRuntime(
        runner,
        max_num_seqs=max_num_seqs,
        max_tokens_per_step=max_tokens_per_step,
        prefill_chunk=prefill_chunk,
        debug_metrics=True,
    )
    results: List[Result] = []
    lock = threading.Lock()
    stopping_template = (
        {"eos_token_id": tokenizer.eos_token_id}
        if use_eos_stop
        else {"eos_token_id": None}
    )
    stop_list = sorted(stop_tokens_override) if stop_tokens_override else []
    stopping_template["stop_token_ids"] = stop_list
    closed_arrivals = (
        closed_arrival_schedule(len(prompts), arrival_mode, lmbda)
        if not open_loop
        else None
    )
    start = time.perf_counter()
    request_seq = itertools.count()
    scheduled_count = 0
    completed_count = 0
    inflight = 0
    total_tokens_emitted = 0
    target_tokens = (
        tokens_target if (open_loop and tokens_target and tokens_target > 0) else None
    )
    if open_loop and target_tokens is None:
        target_tokens = max_tokens * max_num_seqs * 10

    def submit(prompt_idx, arrival_s):
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= arrival_s:
                break
            time.sleep(min(0.001, arrival_s - elapsed))

        prompt = prompts[prompt_idx % len(prompts)]
        prompt_tokens = tokenizer.encode(prompt)
        stopping_settings = dict(stopping_template)

        submit_ns = time.perf_counter_ns()
        _, generator = runtime.submit_request(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            sampler_settings=sampler_settings,
            stopping_settings=stopping_settings,
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )

        first_ns = None
        finish_ns = submit_ns
        final_tokens = 0
        emit_trace: List[Dict[str, object]] = []
        try:
            for response in generator:
                if first_ns is None and response.generation_tokens > 0:
                    first_ns = time.perf_counter_ns()
                final_tokens = response.generation_tokens
                emit_trace.append(
                    {
                        "tokens": int(response.generation_tokens),
                        "token_id": int(getattr(response, "token", 0)),
                        "finish_reason": response.finish_reason,
                    }
                )
                if response.finish_reason:
                    finish_ns = time.perf_counter_ns()
                    break
        except Exception:
            finish_ns = time.perf_counter_ns()
        finally:
            if finish_ns is None:
                finish_ns = time.perf_counter_ns()
            with lock:
                nonlocal completed_count, inflight, total_tokens_emitted
                results.append(
                    Result(
                        submit_ns=submit_ns,
                        first_token_ns=first_ns,
                        finish_ns=finish_ns,
                        gen_tokens=max(final_tokens, 0),
                        emit_trace=list(emit_trace),
                    )
                )
                if EMIT_TRACE:
                    BENCH_LOG.info(
                        "emit_trace prompt_idx=%s events=%s", prompt_idx, emit_trace
                    )
                completed_count += 1
                inflight = max(inflight - 1, 0)
                total_tokens_emitted += max(final_tokens, 0)
                if BENCH_TRACE:
                    BENCH_LOG.info(
                        "open_loop.complete prompt_idx=%s tokens=%s inflight=%s backlog=%s emitted=%s",
                        prompt_idx,
                        final_tokens,
                        inflight,
                        scheduled_count - completed_count,
                        total_tokens_emitted,
                    )
                should_schedule = False
                backlog = scheduled_count - completed_count
                should_schedule = should_schedule_open_loop(
                    backlog,
                    target_tokens=target_tokens,
                    total_tokens_emitted=total_tokens_emitted,
                    backlog_limit=open_loop_backlog_limit,
                )
                if not should_schedule:
                    runner.open_loop_draining = True
            if should_schedule:
                schedule_next()

    threads = []
    arrival_lock = threading.Lock()

    def schedule_next() -> bool:
        nonlocal scheduled_count, inflight
        with arrival_lock:
            if not open_loop and scheduled_count >= len(prompts):
                return False
            arrival_s = time.perf_counter() - start
            if open_loop:
                backlog = scheduled_count - completed_count
                if not should_schedule_open_loop(
                    backlog,
                    target_tokens=target_tokens,
                    total_tokens_emitted=total_tokens_emitted,
                    backlog_limit=open_loop_backlog_limit,
                ):
                    return False
            else:
                arrival_s = closed_arrivals[scheduled_count]
            prompt_idx = next(request_seq)
            scheduled_count += 1
            inflight += 1
            if BENCH_TRACE:
                BENCH_LOG.info(
                    "open_loop.schedule idx=%s backlog=%s limit=%s targets=%s/%s",
                    prompt_idx,
                    scheduled_count - completed_count,
                    open_loop_backlog_limit,
                    total_tokens_emitted,
                    target_tokens,
                )
        th = threading.Thread(target=submit, args=(prompt_idx, arrival_s), daemon=True)
        th.start()
        threads.append(th)
        return True

    if open_loop:
        initial_requests = max(max_num_seqs, open_loop_backlog_limit or max_num_seqs)
    else:
        initial_requests = len(prompts)

    for _ in range(max(1, initial_requests)):
        if not schedule_next():
            break

    while True:
        with lock:
            done = completed_count >= scheduled_count and inflight == 0
        if done:
            break
        time.sleep(0.01)

    for th in threads:
        th.join()

    runtime.shutdown()
    history = runtime.metrics_history()
    return results, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="mlx-community/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--mode",
        choices=("fixed", "eos"),
        default="fixed",
        help="fixed: force max_tokens, eos: stop when EOS tokens reached.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--ttft-window",
        type=int,
        default=0,
        help="Earliest completion count to include when reporting TTFT window stats (default: max_num_seqs).",
    )
    parser.add_argument(
        "--static-mode",
        choices=("auto", "batch", "runtime"),
        default="auto",
        help="How to measure the static baseline: 'batch' uses batch_generate, 'runtime' reuses the server runtime, 'auto' chooses runtime whenever a paged decode engine is selected.",
    )
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument(
        "--steady-fraction",
        type=float,
        default=0.8,
        help="Fraction of max_num_seqs considered steady-state for TPS reporting.",
    )
    parser.add_argument(
        "--stop-token",
        type=int,
        action="append",
        default=None,
        help="Explicit stop token id (repeat for multiple). Defaults to tokenizer eos ids.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help="Maximum concurrent sequences for the continuous runtime.",
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=256,
        help="Prefill chunk size for the continuous runtime.",
    )
    parser.add_argument(
        "--prefill-ramp-chunk",
        type=int,
        default=64,
        help="Prefill chunk used for the first slice of each prompt (0 disables ramp).",
    )
    parser.add_argument(
        "--prefill-hybrid-threshold",
        type=int,
        default=0,
        help="Number of initial prompt tokens to run through dense prefill before paging (0 disables).",
    )
    parser.add_argument(
        "--prefill-ramp-budget-ms",
        type=float,
        default=150.0,
        help="Target millisecond budget for the first paged prefill slice (<=0 disables adaptation).",
    )
    parser.add_argument(
        "--prefill-profile",
        action="store_true",
        help="Collect a single prefill chunk profile and embed metrics in the summary JSON.",
    )
    parser.add_argument(
        "--max-tokens-per-step",
        type=int,
        default=4096,
        help="Token budget per scheduler step for the continuous runtime.",
    )
    parser.add_argument(
        "--decode-unroll",
        type=int,
        default=4,
        help="Number of decode iterations to unroll per scheduler tick.",
    )
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help="Keep scheduling new requests to maintain occupancy until the token target is reached.",
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("burst", "poisson"),
        default="burst",
        help="Arrival distribution for closed-loop runs (ignored whenever --open-loop is set).",
    )
    parser.add_argument(
        "--tokens-target",
        type=int,
        default=0,
        help="Total generation tokens to emit in open-loop mode (default: 10x max_num_seqs*max_tokens).",
    )
    parser.add_argument(
        "--open-loop-backlog-factor",
        type=float,
        default=4.0,
        help="Multiplier for max_num_seqs to cap outstanding open-loop requests (<=0 disables the cap).",
    )
    parser.add_argument(
        "--attn-backend",
        choices=("auto", "dense", "paged"),
        default="auto",
        help="Attention backend passed to ModelRunner (auto enables paged when supported).",
    )
    parser.add_argument(
        "--decode-engine",
        choices=("dense", "paged", "paged-arrays", "paged-arrays+compile"),
        default="paged-arrays",
        help="Decode engine flag passed to the server runtime.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path to write benchmark summary JSON (default: benchmarks/bench_summary_<decode>.json).",
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=16,
        help="Paged KV block size when paged attention is active.",
    )
    parser.add_argument(
        "--kv-pool-blocks",
        default="auto",
        help="Total paged KV blocks ('auto' derives from Metal working-set).",
    )
    parser.add_argument(
        "--kv-quant",
        choices=("none", "int4_v"),
        default="none",
        help="Optional KV quantization mode for paged attention (int4_v only).",
    )
    parser.add_argument(
        "--kv-quant-group-size",
        type=int,
        default=64,
        help="Group size used for paged KV quantization.",
    )
    parser.add_argument(
        "--paged-vec-width",
        type=int,
        default=None,
        help="Optional vector width override for paged attention kernels.",
    )
    parser.add_argument(
        "--paged-threads-per-head",
        type=int,
        default=None,
        help="Optional threads-per-head override for paged attention kernels.",
    )
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=1.5,
        help="Minimum continuous/static tokens/s ratio required when concurrency>=4 (<=0 disables).",
    )
    parser.add_argument(
        "--min-apc-hit-rate",
        type=float,
        default=0.2,
        help="Minimum APC hit rate required when lookups>0 (<=0 disables).",
    )
    args = parser.parse_args()

    summary_path = args.summary_path
    if summary_path:
        summary_path = Path(summary_path)
    else:
        default_dir = Path("benchmarks")
        default_name = f"bench_summary_{args.decode_engine.replace('/', '_')}_{args.attn_backend}.json"
        summary_path = default_dir / default_name

    model, tokenizer = load(path_or_hf_repo=args.repo)
    n_reqs = normalise_request_count(args.n, args.concurrency)
    base_unit = "Tell me a haiku about mac GPUs. "
    repeats = max(1, (args.prompt_len // len(base_unit)) + 1)
    prompt_text = (base_unit * repeats)[: args.prompt_len]
    prompts = [prompt_text for _ in range(n_reqs)]
    max_num_seqs, kv_meta = estimate_safe_max_num_seqs(
        model,
        args.max_num_seqs,
        args.max_tokens,
        limit_bytes=KV_SLAB_LIMIT_BYTES,
    )
    if kv_meta is not None:
        logging.warning(
            "max_num_seqs reduced from %s to %s based on estimated KV slab size %.2f MB per slot (limit %.2f MB)",
            args.max_num_seqs,
            max_num_seqs,
            kv_meta["per_slot_bytes"] / (1024 * 1024),
            kv_meta["limit_bytes"] / (1024 * 1024),
        )
    target_tokens = None
    if args.open_loop:
        default_budget = max_num_seqs * args.max_tokens * 10
        target_tokens = args.tokens_target if args.tokens_target > 0 else default_budget

    explicit_stops = set(args.stop_token) if args.stop_token else None
    controls = select_token_controls(
        mode=args.mode,
        tokenizer=tokenizer,
        explicit_stops=explicit_stops,
    )
    ttft_window = (
        args.ttft_window
        if args.ttft_window and args.ttft_window > 0
        else max(1, min(args.concurrency, max_num_seqs))
    )
    kv_pool_blocks_arg = args.kv_pool_blocks
    if (
        isinstance(kv_pool_blocks_arg, str)
        and kv_pool_blocks_arg.strip().lower() == "auto"
    ):
        kv_pool_blocks_arg = None
    elif kv_pool_blocks_arg is not None:
        kv_pool_blocks_arg = int(kv_pool_blocks_arg)

    est_service = max(args.max_tokens / 200.0, 0.01)
    lmbda = max(0.1, args.concurrency / est_service)
    runtime_static_enabled = should_use_runtime_static(
        args.decode_engine, args.static_mode
    )
    if runtime_static_enabled:
        static_results = run_static_runtime(
            model,
            tokenizer,
            prompts,
            args.max_tokens,
            lmbda,
            max_num_seqs=max_num_seqs,
            prefill_chunk=args.prefill_chunk,
            prefill_ramp_chunk=args.prefill_ramp_chunk,
            max_tokens_per_step=args.max_tokens_per_step,
            decode_unroll=args.decode_unroll,
            stop_tokens_override=controls.runtime_stop_tokens,
            use_eos_stop=controls.use_eos_stop,
            sampler_settings=controls.sampler_settings,
            attn_backend=args.attn_backend,
            kv_block_size=args.kv_block_size,
            kv_pool_blocks=kv_pool_blocks_arg,
            paged_vec_width=args.paged_vec_width,
            paged_threads_per_head=args.paged_threads_per_head,
            kv_quant_mode=args.kv_quant,
            kv_quant_group_size=args.kv_quant_group_size,
            decode_engine=args.decode_engine,
            prefill_hybrid_threshold=args.prefill_hybrid_threshold,
            prefill_ramp_budget_ms=args.prefill_ramp_budget_ms,
        )
    else:
        static_results = run_static(
            model,
            tokenizer,
            prompts,
            args.max_tokens,
            stop_tokens=controls.static_stop_tokens,
            sampler_settings=controls.sampler_settings,
            open_loop=args.open_loop,
            tokens_target=target_tokens,
        )
    static_mode_effective = "runtime" if runtime_static_enabled else "batch"
    static_summary = summarize(
        "static_batch_generate", static_results, ttft_window=ttft_window
    )
    static_summary.update(emit_trace_summary(static_results))

    backlog_limit = None
    if args.open_loop and args.open_loop_backlog_factor > 0:
        backlog_limit = max(
            int(math.ceil(args.open_loop_backlog_factor * args.max_num_seqs)),
            args.max_num_seqs,
        )

    continuous_results, history = run_continuous(
        model,
        tokenizer,
        prompts,
        args.max_tokens,
        lmbda,
        max_num_seqs=max_num_seqs,
        prefill_chunk=args.prefill_chunk,
        prefill_ramp_chunk=args.prefill_ramp_chunk,
        max_tokens_per_step=args.max_tokens_per_step,
        decode_unroll=args.decode_unroll,
        stop_tokens_override=controls.runtime_stop_tokens,
        use_eos_stop=controls.use_eos_stop,
        sampler_settings=controls.sampler_settings,
        open_loop=args.open_loop,
        tokens_target=target_tokens,
        attn_backend=args.attn_backend,
        kv_block_size=args.kv_block_size,
        kv_pool_blocks=kv_pool_blocks_arg,
        paged_vec_width=args.paged_vec_width,
        paged_threads_per_head=args.paged_threads_per_head,
        kv_quant_mode=args.kv_quant,
        kv_quant_group_size=args.kv_quant_group_size,
        open_loop_backlog_limit=backlog_limit,
        decode_engine=args.decode_engine,
        prefill_hybrid_threshold=args.prefill_hybrid_threshold,
        prefill_ramp_budget_ms=args.prefill_ramp_budget_ms,
        arrival_mode=args.arrival_mode,
    )
    continuous_summary = summarize(
        "continuous_runtime", continuous_results, ttft_window=ttft_window
    )
    continuous_summary.update(emit_trace_summary(continuous_results))
    steady_tps = compute_steady_state_tps(
        history,
        steady_fraction=args.steady_fraction,
        max_num_seqs=max_num_seqs,
    )
    phases = phase_summaries(
        history,
        steady_fraction=args.steady_fraction,
        max_num_seqs=max_num_seqs,
    )
    apc_stats = extract_apc_stats(history)
    compile_stats = extract_compile_stats(history)
    prefill_profile = None
    if args.prefill_profile:
        prefill_profile = collect_prefill_profile_metrics(
            model=model,
            chunk_len=args.prefill_chunk,
            ramp_chunk=args.prefill_ramp_chunk,
        )
        if prefill_profile:
            BENCH_LOG.info(
                "prefill_profile: chunk=%s attn=%.3fs mlp=%.3fs overlay=%.3fs",
                prefill_profile["chunk_len"],
                prefill_profile["array_prefill_attn_s"],
                prefill_profile["array_prefill_mlp_s"],
                prefill_profile["array_prefill_overlay_s"],
            )
    history_path = None
    if args.summary_path:
        history_path = Path(args.summary_path).with_suffix(".history.json")
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(history, indent=2))
        except Exception as exc:
            logging.warning(
                "Unable to write runtime history to %s: %s", history_path, exc
            )
    if steady_tps:
        print(
            f"[continuous_runtime steady] active>={args.steady_fraction * max_num_seqs:.1f} "
            f"tokens/s={steady_tps:.2f}"
        )
    if phases:
        for phase_name in ("ramp", "steady", "tail"):
            stats = phases.get(phase_name)
            if not stats:
                continue
            print(
                f"[phase {phase_name}] ticks={stats['ticks']} time={stats['time_s']:.2f}s "
                f"tokens={stats['tokens']} tokens/s={stats['tokens_per_sec']:.2f} "
                f"mean_decode_ms={stats['mean_decode_ms']:.1f} model_ms={stats['decode_model_ms']:.1f} "
                f"driver_ms={stats['decode_driver_ms']:.1f} mean_batch={stats['mean_batch']:.1f} "
                f"prefill_tokens={stats['prefill_tokens']} prefill_wall_ms={stats['prefill_wall_s'] * 1000.0:.1f}"
            )
        spikes = phases.get("spikes")
        if spikes:
            print(
                f"[phase spikes] count={spikes['count']} time={spikes['time_s']:.2f}s "
                f"threshold={spikes['threshold_s']:.2f}s"
            )
    if apc_stats:
        print(
            "[apc] hits={hits:.0f} lookups={lookups:.0f} hit_rate={hit_rate:.2%} tokens_reused={tokens:.0f}".format(
                hits=apc_stats.get("prefix_hits", 0.0),
                lookups=apc_stats.get("prefix_lookups", 0.0),
                hit_rate=apc_stats.get("prefix_hit_rate", 0.0),
                tokens=apc_stats.get("prefix_tokens_reused", 0.0),
            )
        )
    if compile_stats:
        parts = []
        for key in (
            "compile_cache_hits",
            "compile_cache_misses",
            "array_phase1_compile_hits",
            "array_phase1_compile_misses",
        ):
            if key in compile_stats:
                parts.append(f"{key}={compile_stats[key]:.0f}")
        if "array_phase1_duration_s" in compile_stats:
            dur_ms = compile_stats["array_phase1_duration_s"] * 1000.0
            parts.append(f"array_phase1_duration_ms={dur_ms:.2f}")
        if parts:
            print("[compile_stats] " + " ".join(parts))

    kill_reasons = _maybe_kill_switch(
        args=args,
        static_summary=static_summary,
        continuous_summary=continuous_summary,
        apc_stats=apc_stats,
    )

    summary_payload = {
        "decode_engine": args.decode_engine,
        "prefill_hybrid_threshold": args.prefill_hybrid_threshold,
        "static_mode": static_mode_effective,
        "static_summary": static_summary,
        "continuous_summary": continuous_summary,
        "continuous_steady_tokens_per_sec": steady_tps,
        "continuous_phase_details": phases,
        "effective_max_num_seqs": max_num_seqs,
        "kv_capacity_meta": kv_meta,
        "continuous_apc_stats": apc_stats,
        "runtime_compile_stats": compile_stats,
        "kill_switch": {"tripped": bool(kill_reasons), "reasons": kill_reasons},
    }
    if prefill_profile:
        summary_payload["prefill_profile"] = prefill_profile
    summary_text = json.dumps(summary_payload, indent=2)
    print(summary_text)
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        logging.info("Wrote bench summary to %s", summary_path)
    except Exception:
        logging.warning(
            "Unable to write bench summary to %s", summary_path, exc_info=True
        )

    static_tps = static_summary.get("tokens_per_sec", 0.0)
    cont_tps = continuous_summary.get("tokens_per_sec", 0.0)
    throughput_ratio = (cont_tps / static_tps) if static_tps > 0 else float("inf")
    if (
        args.min_throughput_ratio > 0
        and args.concurrency >= 4
        and static_tps > 0
        and throughput_ratio < args.min_throughput_ratio
    ):
        print(
            f"WARNING: continuous tokens/s {continuous_summary['tokens_per_sec']:.2f} "
            f"< {args.min_throughput_ratio:.2f}x static {static_summary['tokens_per_sec']:.2f}"
        )

    if kill_reasons:
        raise SystemExit("Kill-switch triggered: " + "; ".join(kill_reasons))


if __name__ == "__main__":
    main()
