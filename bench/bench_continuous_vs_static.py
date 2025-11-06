# ABOUTME: Benchmarks static batch_generate against continuous batching runtime.
# ABOUTME: Uses wall-clock throughput under Poisson arrivals to track gains.

"""
Benchmark static batch_generate vs continuous batching runtime under Poisson arrivals.
Run: python bench/bench_continuous_vs_static.py --repo mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import List, Optional

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.server_batched.engine import ModelRunner
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime

random.seed(0)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class Result:
    submit_ns: int
    first_token_ns: Optional[int]
    finish_ns: int
    gen_tokens: int


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


def run_static(model, tokenizer, prompts, max_tokens):
    prompt_token_batches = [tokenizer.encode(p) for p in prompts]
    start_ns = time.perf_counter_ns()
    response = batch_generate(
        model,
        tokenizer,
        prompt_token_batches,
        max_tokens=max_tokens,
        verbose=False,
    )
    end_ns = time.perf_counter_ns()
    results = []
    for text in response.texts:
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        results.append(
            Result(
                submit_ns=start_ns,
                first_token_ns=end_ns,
                finish_ns=end_ns,
                gen_tokens=tokens,
            )
        )
    return results


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
):
    runner = ModelRunner(
        model,
        tokenizer,
        draft_model=None,
        max_num_seqs=max_num_seqs,
        prefill_chunk=prefill_chunk,
    )
    runtime = ContinuousBatchingRuntime(
        runner,
        max_num_seqs=max_num_seqs,
        max_tokens_per_step=max_tokens_per_step,
        prefill_chunk=prefill_chunk,
        debug_metrics=True,
    )
    results: List[Result] = []
    lock = threading.Lock()
    arrivals = poisson_arrivals(lmbda, len(prompts))
    start = time.perf_counter()

    def submit(idx, arrival_s):
        stats = runtime.runner.collect_step_stats()
        metrics = runtime.scheduler.metrics
        logging.info("[BENCH] tick stats decode_ms=%.3f decode_tokens=%s prefill_tokens=%s active=%s wait=%s" % (
            stats.get('decode_duration_s', 0.0) * 1000.0,
            stats.get('decode_tokens'),
            stats.get('prefill_tokens'),
            metrics.get('active_sequences'),
            metrics.get('wait_queue_depth'),
        ))
        # Wait until scheduled arrival time
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= arrival_s:
                break
            time.sleep(min(0.001, arrival_s - elapsed))

        prompt = prompts[idx]
        prompt_tokens = tokenizer.encode(prompt)
        sampler_settings = {
            "temp": 0.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": 0,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.0,
            "xtc_special_tokens": [tokenizer.eos_token_id, tokenizer.encode("\n")],
        }
        stopping_settings = {"eos_token_id": tokenizer.eos_token_id}

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
                results.append(
                    Result(
                        submit_ns=submit_ns,
                        first_token_ns=first_ns,
                        finish_ns=finish_ns,
                        gen_tokens=max(final_tokens, 0),
                    )
                )

    threads = []
    for idx, arrival in enumerate(arrivals):
        th = threading.Thread(target=submit, args=(idx, arrival), daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    runtime.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help="Maximum concurrent sequences for the continuous runtime.",
    )
    parser.add_argument(
        "--prefill_chunk",
        type=int,
        default=1024,
        help="Prefill chunk size for the continuous runtime.",
    )
    parser.add_argument(
        "--max_tokens_per_step",
        type=int,
        default=4096,
        help="Token budget per scheduler step for the continuous runtime.",
    )
    args = parser.parse_args()

    model, tokenizer = load(path_or_hf_repo=args.repo)
    base_prompt = "Tell me a haiku about mac GPUs." * 4
    prompts = [base_prompt[: args.prompt_len] for _ in range(args.n)]

    static_results = run_static(model, tokenizer, prompts, args.max_tokens)
    static_tps, static_wall, static_tokens = summarize("static_batch_generate", static_results)

    est_service = max(args.max_tokens / 200.0, 0.01)
    lmbda = max(0.1, args.concurrency / est_service)
    continuous_results = run_continuous(
        model,
        tokenizer,
        prompts,
        args.max_tokens,
        lmbda,
        max_num_seqs=args.max_num_seqs,
        prefill_chunk=args.prefill_chunk,
        max_tokens_per_step=args.max_tokens_per_step,
    )
    cont_tps, cont_wall, cont_tokens = summarize("continuous_runtime", continuous_results)

    print(
        json.dumps(
            {
                "static_tokens_per_sec": static_tps,
                "static_total_tokens": static_tokens,
                "static_wall_seconds": static_wall,
                "continuous_tokens_per_sec": cont_tps,
                "continuous_total_tokens": cont_tokens,
                "continuous_wall_seconds": cont_wall,
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
