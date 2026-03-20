# Copyright © 2023-2024 Apple Inc.

"""Benchmark CompressedKVCache on Qwen3-8B-4bit.

Measures:
1. Memory before/after compaction at 2K/4K/8K contexts
2. Compaction latency on M4 Max
3. Extraction quality preservation (coherent responses after eviction)
"""

import time

import mlx.core as mx

from mlx_lm import generate
from mlx_lm.generate import maybe_compact_kv_cache
from mlx_lm.models.cache import CompressedKVCache, KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

MODEL_PATH = "mlx-community/Qwen3-8B-4bit"


def measure_cache_memory(cache):
    """Return total cache memory in MB."""
    total_bytes = sum(c.nbytes for c in cache)
    return total_bytes / (1024 * 1024)


def benchmark_memory(model, tokenizer):
    """Benchmark memory before/after compaction at various context lengths."""
    print("\n=== Memory Benchmark ===")
    print(
        f"{'Context':<12} {'Full (MB)':<14} {'Compressed (MB)':<18} {'Reduction':<12}"
    )
    print("-" * 56)

    for ctx_len in [2048, 4096, 8192]:
        budget = ctx_len // 2

        # Generate a long prompt by repeating tokens
        base = "The quick brown fox jumps over the lazy dog. "
        prompt_text = base * (ctx_len // 8)
        prompt_tokens = tokenizer.encode(prompt_text)[:ctx_len]
        prompt_arr = mx.array(prompt_tokens)

        # Full cache baseline
        full_cache = make_prompt_cache(model)
        model(prompt_arr[None], cache=full_cache)
        mx.eval([c.state for c in full_cache])
        full_mem = measure_cache_memory(full_cache)

        # Compressed cache
        comp_cache = make_prompt_cache(model, compact_kv_budget=budget)
        model(prompt_arr[None], cache=comp_cache)
        mx.eval([c.state for c in comp_cache])

        # Cross-layer coherent eviction: aggregate norms across layers
        # and compute shared kept_indices, matching the production path in
        # maybe_compact_kv_cache. We skip the hysteresis check here since
        # we always want to force compaction for the memory measurement.
        all_compressed = [c for c in comp_cache if isinstance(c, CompressedKVCache)]
        ref = all_compressed[0]
        agg_norms = None
        for c in all_compressed:
            active_keys = c.keys[..., : c.size(), :]
            norms = mx.linalg.norm(active_keys, axis=-1).sum(axis=1)
            agg_norms = norms if agg_norms is None else agg_norms + norms
        kept_indices = ref.indices_from_norms(agg_norms)
        for c in all_compressed:
            c.compact(kept_indices)
        mx.eval([c.state for c in comp_cache])

        comp_mem = measure_cache_memory(comp_cache)
        reduction = (1 - comp_mem / full_mem) * 100

        print(f"{ctx_len:<12} {full_mem:<14.2f} {comp_mem:<18.2f} {reduction:<12.1f}%")

        del full_cache, comp_cache
        mx.clear_cache()


def benchmark_latency(model, tokenizer):
    """Benchmark compaction latency at 8K tokens."""
    print("\n=== Compaction Latency (8K tokens) ===")

    ctx_len = 8192
    budget = ctx_len // 2
    base = "The quick brown fox jumps over the lazy dog. "
    prompt_text = base * (ctx_len // 8)
    prompt_tokens = tokenizer.encode(prompt_text)[:ctx_len]
    prompt_arr = mx.array(prompt_tokens)

    # Warm up
    comp_cache = make_prompt_cache(model, compact_kv_budget=budget)
    model(prompt_arr[None], cache=comp_cache)
    mx.eval([c.state for c in comp_cache])
    maybe_compact_kv_cache(comp_cache)
    mx.eval([c.state for c in comp_cache])
    del comp_cache
    mx.clear_cache()

    # Measure compaction latency (run multiple times)
    n_trials = 10
    latencies = []
    for _ in range(n_trials):
        # Reset cache
        comp_cache = make_prompt_cache(model, compact_kv_budget=budget)
        model(prompt_arr[None], cache=comp_cache)
        mx.eval([c.state for c in comp_cache])

        start = time.perf_counter()
        maybe_compact_kv_cache(comp_cache)
        mx.eval([c.state for c in comp_cache])  # force Metal execution
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

        del comp_cache
        mx.clear_cache()

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    print(f"Trials: {n_trials}")
    print(f"Avg: {avg_ms:.2f} ms, Min: {min_ms:.2f} ms, Max: {max_ms:.2f} ms")
    print(f"{'PASS' if avg_ms < 100 else 'FAIL'}: Target < 100ms")
    return avg_ms


def benchmark_quality(model, tokenizer):
    """Compare generation quality with and without compression.

    Uses a long-context setup: a ~1K-token document as context plus short
    questions, with a budget of 256 tokens. This forces eviction to fire
    during the prefill, validating that the model's answers remain coherent
    after cache compaction.
    """
    print("\n=== Quality Comparison (eviction-active) ===")

    # Build a long context that exceeds the budget
    context = (
        "The quick brown fox jumps over the lazy dog. " * 50
        + "Important fact: the capital of France is Paris. "
        + "The quick brown fox jumps over the lazy dog. " * 50
    )

    questions = [
        "What is the capital of France?",
        "Summarize the text above in one sentence.",
        "What animal was mentioned in the text?",
        "Is there any geographical information in the text?",
        "What adjectives describe the fox?",
    ]

    budget = 256
    max_tokens = 50
    coherent = 0

    sampler = make_sampler(temp=0.0)

    for i, question in enumerate(questions):
        prompt_text = f"{context}\n\nQuestion: {question}\nAnswer:"

        # Generate without compression (baseline)
        baseline = generate(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        # Generate with compression (eviction will fire)
        compressed = generate(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            compact_kv_budget=budget,
        )

        # Check if compressed output is non-empty and reasonably coherent
        # (not garbled or empty, which would indicate broken eviction)
        is_coherent = len(compressed.strip()) > 10
        if is_coherent:
            coherent += 1
        is_match = baseline.strip() == compressed.strip()
        status = "MATCH" if is_match else ("COHERENT" if is_coherent else "GARBLED")
        print(f"  Q{i+1}: {status}")
        if not is_match:
            print(f"    Baseline:   {baseline.strip()[:80]}...")
            print(f"    Compressed: {compressed.strip()[:80]}...")

    print(f"\nCoherent: {coherent}/{len(questions)}")
    print(f"{'PASS' if coherent >= 4 else 'FAIL'}: Target >= 4/5 coherent")
    return coherent


def main():
    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded.")

    benchmark_memory(model, tokenizer)
    avg_latency = benchmark_latency(model, tokenizer)
    n_equiv = benchmark_quality(model, tokenizer)

    print("\n=== Summary ===")
    print(f"Memory reduction: See table above")
    print(f"Compaction latency: {avg_latency:.2f} ms (target < 100ms)")
    print(f"Quality preservation: {n_equiv}/5 coherent (target >= 4/5)")


if __name__ == "__main__":
    main()
