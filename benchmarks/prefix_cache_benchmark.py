import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.utils import load


def encode(tokenizer, text):
    return list(tokenizer.encode(text))


def cache_nbytes(prompt_cache):
    return int(sum(getattr(c, "nbytes", 0) for c in prompt_cache))


def cache_logical_length(prompt_cache):
    lengths = [c.logical_length() for c in prompt_cache if hasattr(c, "logical_length")]
    return max(lengths) if lengths else None


def prefill_snapshot(model, tokens, prefill_step_size):
    if not tokens:
        raise ValueError("Cannot prefill an empty token sequence")
    prompt_cache = make_prompt_cache(model)
    token_array = mx.array(tokens)
    start = time.perf_counter()
    for offset in range(0, len(tokens), prefill_step_size):
        chunk = token_array[offset : offset + prefill_step_size]
        model(chunk[None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        mx.clear_cache()
    prefill_s = time.perf_counter() - start
    return prompt_cache, prefill_s


def run_generation(model, tokenizer, tokens, prompt_cache, max_tokens, prefill_step_size):
    if not tokens:
        raise ValueError("Cannot generate from an empty token tail")
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    first_s = None
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
    ):
        if first_s is None:
            first_s = time.perf_counter() - start
        last = response
    total_s = time.perf_counter() - start
    mx.eval([c.state for c in prompt_cache])
    return {
        "ttft_s": None if first_s is None else first_s,
        "total_s": total_s,
        "generated_tokens": 0 if last is None else int(last.generation_tokens),
        "prompt_tps": None if last is None else float(last.prompt_tps),
        "generation_tps": None if last is None else float(last.generation_tps),
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9,
    }


def run_cold(model, tokenizer, tokens, max_tokens, prefill_step_size):
    prompt_cache = make_prompt_cache(model)
    result = run_generation(
        model,
        tokenizer,
        tokens,
        prompt_cache,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
    )
    return {
        "name": "cold_full_prefill",
        "cache_hit_kind": "cold",
        "full_prompt_tokens": len(tokens),
        "cached_tokens": 0,
        "replayed_tokens": len(tokens),
        "cache_bytes": cache_nbytes(prompt_cache),
        "cache_logical_length": cache_logical_length(prompt_cache),
        **result,
    }


def insert_snapshot(lru, model_key, key_tokens, prompt_cache, prefill_s):
    lru.insert_cache(model_key, list(key_tokens), prompt_cache)
    return {
        "prefill_snapshot_tokens": len(key_tokens),
        "inserted_key_tokens": len(key_tokens),
        "cache_logical_length": cache_logical_length(prompt_cache),
        "cache_bytes": cache_nbytes(prompt_cache),
        "prefill_s": prefill_s,
    }


def run_hot(name, cache_hit_kind, lru, model_key, model, tokenizer, tokens, max_tokens, prefill_step_size):
    prompt_cache, rest = lru.fetch_nearest_cache(model_key, list(tokens))
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
        rest = list(tokens)
    result = run_generation(
        model,
        tokenizer,
        rest,
        prompt_cache,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
    )
    return {
        "name": name,
        "cache_hit_kind": cache_hit_kind,
        "full_prompt_tokens": len(tokens),
        "cached_tokens": len(tokens) - len(rest),
        "replayed_tokens": len(rest),
        "cache_bytes": cache_nbytes(prompt_cache),
        "cache_logical_length": cache_logical_length(prompt_cache),
        **result,
    }


def build_prompts(tokenizer, prompt, target_prefix_tokens):
    if prompt:
        shared_prefix = prompt
    else:
        sentence = (
            "Apple Silicon inference note: MLX uses unified memory and Metal kernels. "
            "Prefix cache reuse matters for long multi-turn assistant workloads because shared system prompts, tools, and retrieved context should not be prefetched again. "
            "This benchmark repeats technical context so prefill dominates a smoke-sized prompt. "
        )
        blocks = []
        for index in range(1, 10000):
            blocks.append(f"Context block {index}: {sentence}")
            shared_prefix = "\n".join(blocks)
            if len(encode(tokenizer, shared_prefix)) >= target_prefix_tokens:
                break
    suffix_a = "\nTask A: Produce a concise checklist for validating prefix-cache behavior on Apple Silicon. Include tests, profiling, and benchmark evidence.\nAnswer:"
    suffix_b = "\nTask B: Produce a concise risk assessment for changing prefix-cache restore semantics. Include correctness hazards and benchmark requirements.\nAnswer:"
    return shared_prefix, shared_prefix + suffix_a, shared_prefix + suffix_b


def rounded(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {key: rounded(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rounded(value) for value in obj]
    return obj


def emit(event, **payload):
    print(json.dumps(rounded({"event": event, **payload}), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--target-prefix-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo), "branch", "--show-current"], text=True
        ).strip()
    except Exception:
        commit = None
        branch = None

    emit(
        "env",
        branch=branch,
        commit=commit,
        model=args.model,
        platform=platform.platform(),
        machine=platform.machine(),
    )

    start = time.perf_counter()
    model, tokenizer = load(args.model)
    emit("loaded", load_s=time.perf_counter() - start)

    shared_prefix, prompt_a, prompt_b = build_prompts(
        tokenizer, args.prompt, args.target_prefix_tokens
    )
    prefix_tokens = encode(tokenizer, shared_prefix)
    tokens_a = encode(tokenizer, prompt_a)
    tokens_b = encode(tokenizer, prompt_b)
    emit(
        "prompt",
        shared_prefix_tokens=len(prefix_tokens),
        prompt_a_tokens=len(tokens_a),
        prompt_b_tokens=len(tokens_b),
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
    )

    warm_cache = make_prompt_cache(model)
    for _ in stream_generate(
        model,
        tokenizer,
        encode(tokenizer, "Warmup prompt for kernel compilation. Answer:"),
        max_tokens=8,
        prompt_cache=warm_cache,
        prefill_step_size=args.prefill_step_size,
    ):
        pass
    mx.eval([c.state for c in warm_cache])
    mx.clear_cache()

    cold = run_cold(
        model,
        tokenizer,
        tokens_b,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
    )
    emit("result", **cold)

    exact_lru = LRUPromptCache(max_size=4)
    exact_key = tokens_b[:-1]
    exact_cache, exact_prefill_s = prefill_snapshot(
        model, exact_key, args.prefill_step_size
    )
    emit(
        "snapshot",
        name="exact_prefix_seed",
        **insert_snapshot(exact_lru, args.model, exact_key, exact_cache, exact_prefill_s),
    )
    exact_hot = run_hot(
        "exact_prefix_hot_reuse",
        "exact_prefix",
        exact_lru,
        args.model,
        model,
        tokenizer,
        tokens_b,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
    )
    emit("result", **exact_hot)

    nearest_lru = LRUPromptCache(max_size=4)
    nearest_cache, nearest_prefill_s = prefill_snapshot(
        model, tokens_a, args.prefill_step_size
    )
    emit(
        "snapshot",
        name="nearest_prefix_seed",
        **insert_snapshot(nearest_lru, args.model, tokens_a, nearest_cache, nearest_prefill_s),
    )
    nearest_hot = run_hot(
        "nearest_prefix_hot_reuse",
        "nearest_prefix",
        nearest_lru,
        args.model,
        model,
        tokenizer,
        tokens_b,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
    )
    emit("result", **nearest_hot)

    emit(
        "comparison",
        exact_cached_tokens=exact_hot["cached_tokens"],
        exact_replayed_tokens=exact_hot["replayed_tokens"],
        exact_ttft_speedup_vs_cold=(
            cold["ttft_s"] / exact_hot["ttft_s"] if exact_hot["ttft_s"] else None
        ),
        exact_total_speedup_vs_cold=(
            cold["total_s"] / exact_hot["total_s"] if exact_hot["total_s"] else None
        ),
        nearest_cached_tokens=nearest_hot["cached_tokens"],
        nearest_replayed_tokens=nearest_hot["replayed_tokens"],
        nearest_ttft_speedup_vs_cold=(
            cold["ttft_s"] / nearest_hot["ttft_s"] if nearest_hot["ttft_s"] else None
        ),
        nearest_total_speedup_vs_cold=(
            cold["total_s"] / nearest_hot["total_s"] if nearest_hot["total_s"] else None
        ),
    )


if __name__ == "__main__":
    main()
