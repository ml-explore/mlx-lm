# ABOUTME: Benchmarks static batch_generate against continuous batching runtime.
# ABOUTME: Uses wall-clock throughput under Poisson arrivals to track gains.

"""
Benchmark static batch_generate vs continuous batching runtime under Poisson arrivals.
Run: python bench/bench_continuous_vs_static.py --repo mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import itertools
import os
from contextlib import contextmanager
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.server_batched.engine import ModelRunner
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime

random.seed(0)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# TODO: Drop this fallback once paged attention lands and dense slabs disappear.
KV_SLAB_LIMIT_BYTES = int(os.environ.get("MLX_KV_SLAB_LIMIT_MB", "320")) * 1024 * 1024


@dataclass
class Result:
    submit_ns: int
    first_token_ns: Optional[int]
    finish_ns: int
    gen_tokens: int


@dataclass(frozen=True)
class TokenControls:
    sampler_settings: Dict[str, float]
    static_stop_tokens: List[int]
    runtime_stop_tokens: Set[int]
    use_eos_stop: bool


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
            }
        total_time = sum(history[i].get("decode_duration_s", 0.0) for i in indices)
        total_tokens = sum(history[i].get("decode_tokens", 0) for i in indices)
        mean_batch = (
            sum(history[i].get("decode_batch_size", history[i].get("active_sequences", 0)) for i in indices)
            / ticks
        )
        total_prefill = sum(history[i].get("prefill_tokens", 0) for i in indices)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        mean_decode_ms = (total_time / ticks) * 1000.0 if ticks else 0.0
        return {
            "ticks": ticks,
            "time_s": total_time,
            "tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "mean_decode_ms": mean_decode_ms,
            "mean_batch": mean_batch,
            "prefill_tokens": total_prefill,
        }

    summaries = {
        "ramp": aggregate(ramp_indices),
        "steady": aggregate(steady_indices),
        "tail": aggregate(tail_indices),
    }

    spike_ticks = [tick for tick in history if tick.get("decode_duration_s", 0.0) > spike_threshold_s]
    spike_time = sum(tick.get("decode_duration_s", 0.0) for tick in spike_ticks)
    summaries["spikes"] = {
        "count": len(spike_ticks),
        "time_s": spike_time,
        "threshold_s": spike_threshold_s,
    }

    return summaries


def should_schedule_open_loop(
    backlog: int,
    free_slots: int,
    active_slots: int,
    max_num_seqs: int,
    *,
    target_tokens: Optional[int],
    total_tokens_emitted: int,
) -> bool:
    if active_slots >= max_num_seqs:
        return False
    if free_slots <= 0:
        return False
    if backlog >= max_num_seqs:
        return False
    if target_tokens is not None and total_tokens_emitted >= target_tokens:
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


def summarize(name: str, results: List[Result]):
    if not results:
        print(f"[{name}] n=0 tokens=0 wall=0.00s tokens/s=0.00 ttft mean=0.0ms median=0.0ms p95=0.0ms")
        return 0.0, 0.0, 0

    start_ns = min(r.submit_ns for r in results)
    end_ns = max(r.finish_ns for r in results)
    wall_seconds = max((end_ns - start_ns) / 1e9, 1e-9)
    total_tokens = sum(r.gen_tokens for r in results)
    tokens_per_sec = total_tokens / wall_seconds

    ttfts = []
    for r in results:
        first = r.first_token_ns if r.first_token_ns is not None else r.finish_ns
        ttft_ms = max((first - r.submit_ns) / 1e6, 0.0)
        ttfts.append(ttft_ms)

    ttfts.sort()
    ttft_mean = mean(ttfts) if ttfts else 0.0
    ttft_median = median(ttfts) if ttfts else 0.0
    idx = max(int(math.ceil(0.95 * len(ttfts))) - 1, 0) if ttfts else 0
    ttft_p95 = ttfts[idx] if ttfts else 0.0

    mean_tokens = total_tokens / len(results) if results else 0.0
    print(
        f"[{name}] n={len(results)} tokens={total_tokens} wall={wall_seconds:.2f}s "
        f"tokens/s={tokens_per_sec:.2f} "
        f"ttft mean={ttft_mean:.1f}ms median={ttft_median:.1f}ms p95={ttft_p95:.1f}ms "
        f"gen/req mean={mean_tokens:.1f}"
    )
    return tokens_per_sec, wall_seconds, total_tokens


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
    prompt_token_batches = [tokenizer.encode(p) for p in prompts]
    results: List[Result] = []
    total_tokens = 0
    target_tokens = tokens_target if (open_loop and tokens_target and tokens_target > 0) else None
    sampler = make_sampler(**sampler_settings)
    with _override_tokenizer_stops(tokenizer, stop_tokens):
        while True:
            start_ns = time.perf_counter_ns()
            response = batch_generate(
                model,
                tokenizer,
                prompt_token_batches,
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
                    )
                )
            total_tokens += batch_tokens
            if not open_loop:
                break
            if target_tokens is not None and total_tokens >= target_tokens:
                break
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
    max_tokens_per_step: int,
    stop_tokens_override: Optional[set],
    use_eos_stop: bool,
    sampler_settings: Dict[str, float],
    open_loop: bool,
    tokens_target: Optional[int],
):
    runner = ModelRunner(
        model,
        tokenizer,
        draft_model=None,
        max_num_seqs=max_num_seqs,
        prefill_chunk=prefill_chunk,
        stop_tokens=stop_tokens_override,
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
    closed_arrivals = poisson_arrivals(lmbda, len(prompts)) if not open_loop else None
    start = time.perf_counter()
    request_seq = itertools.count()
    scheduled_count = 0
    completed_count = 0
    inflight = 0
    total_tokens_emitted = 0
    target_tokens = tokens_target if (open_loop and tokens_target and tokens_target > 0) else None
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
        try:
            for response in generator:
                if first_ns is None and response.generation_tokens > 0:
                    first_ns = time.perf_counter_ns()
                final_tokens = response.generation_tokens
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
                    )
                )
                completed_count += 1
                inflight = max(inflight - 1, 0)
                total_tokens_emitted += max(final_tokens, 0)
                should_schedule = False
                if open_loop:
                    allocator = getattr(runner, "slot_allocator", None)
                if allocator is not None:
                    free_slots = allocator.available()
                    active_slots = allocator.active_count
                else:
                    free_slots = max(0, max_num_seqs - inflight)
                    active_slots = max_num_seqs - free_slots
                backlog = scheduled_count - completed_count
                should_schedule = should_schedule_open_loop(
                    backlog,
                    free_slots,
                    active_slots,
                    max_num_seqs,
                    target_tokens=target_tokens,
                    total_tokens_emitted=total_tokens_emitted,
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
            if open_loop:
                allocator = getattr(runner, "slot_allocator", None)
                if allocator is not None:
                    free_slots = allocator.available()
                    active_slots = allocator.active_count
                else:
                    free_slots = max(0, max_num_seqs - inflight)
                    active_slots = max_num_seqs - free_slots
                backlog = scheduled_count - completed_count
                if not should_schedule_open_loop(
                    backlog,
                    free_slots,
                    active_slots,
                    max_num_seqs,
                    target_tokens=target_tokens,
                    total_tokens_emitted=total_tokens_emitted,
                ):
                    return False
                arrival_s = time.perf_counter() - start
            else:
                arrival_s = closed_arrivals[scheduled_count]
            prompt_idx = next(request_seq)
            scheduled_count += 1
            inflight += 1
        th = threading.Thread(target=submit, args=(prompt_idx, arrival_s), daemon=True)
        th.start()
        threads.append(th)
        return True

    initial_requests = max_num_seqs if open_loop else len(prompts)

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
    parser.add_argument("--repo", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument(
        "--mode",
        choices=("fixed", "eos"),
        default="fixed",
        help="fixed: force max_tokens, eos: stop when EOS tokens reached.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
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
        default=1024,
        help="Prefill chunk size for the continuous runtime.",
    )
    parser.add_argument(
        "--max-tokens-per-step",
        type=int,
        default=4096,
        help="Token budget per scheduler step for the continuous runtime.",
    )
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help="Keep scheduling new requests to maintain occupancy until the token target is reached.",
    )
    parser.add_argument(
        "--tokens-target",
        type=int,
        default=0,
        help="Total generation tokens to emit in open-loop mode (default: 10x max_num_seqs*max_tokens).",
    )
    args = parser.parse_args()

    model, tokenizer = load(path_or_hf_repo=args.repo)
    n_reqs = normalise_request_count(args.n, args.concurrency)
    base_prompt = "Tell me a haiku about mac GPUs." * 4
    prompts = [base_prompt[: args.prompt_len] for _ in range(n_reqs)]
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
    static_tps, static_wall, static_tokens = summarize("static_batch_generate", static_results)

    est_service = max(args.max_tokens / 200.0, 0.01)
    lmbda = max(0.1, args.concurrency / est_service)
    continuous_results, history = run_continuous(
        model,
        tokenizer,
        prompts,
        args.max_tokens,
        lmbda,
        max_num_seqs=max_num_seqs,
        prefill_chunk=args.prefill_chunk,
        max_tokens_per_step=args.max_tokens_per_step,
        stop_tokens_override=controls.runtime_stop_tokens,
        use_eos_stop=controls.use_eos_stop,
        sampler_settings=controls.sampler_settings,
        open_loop=args.open_loop,
        tokens_target=target_tokens,
    )
    cont_tps, cont_wall, cont_tokens = summarize("continuous_runtime", continuous_results)
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
                f"mean_decode_ms={stats['mean_decode_ms']:.1f} mean_batch={stats['mean_batch']:.1f} "
                f"prefill_tokens={stats['prefill_tokens']}"
            )
        spikes = phases.get("spikes")
        if spikes:
            print(
                f"[phase spikes] count={spikes['count']} time={spikes['time_s']:.2f}s "
                f"threshold={spikes['threshold_s']:.2f}s"
            )

    print(
        json.dumps(
            {
                "static_tokens_per_sec": static_tps,
                "static_total_tokens": static_tokens,
                "static_wall_seconds": static_wall,
                "continuous_tokens_per_sec": cont_tps,
                "continuous_total_tokens": cont_tokens,
                "continuous_wall_seconds": cont_wall,
                "continuous_steady_tokens_per_sec": steady_tps,
                "continuous_phase_details": phases,
                "effective_max_num_seqs": max_num_seqs,
                "kv_capacity_meta": kv_meta,
            },
            indent=2,
        )
    )

    if args.concurrency >= 4 and cont_tps < 1.5 * static_tps:
        raise SystemExit(
            f"FAIL: continuous tokens/s {cont_tps:.2f} < 1.5x static {static_tps:.2f}"
        )


if __name__ == "__main__":
    main()
