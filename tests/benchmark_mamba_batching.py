#!/usr/bin/env python3
"""
Benchmark script for MambaCache batching with prompt caches.

This script tests the new batching capability with hybrid models
like Qwen3-Next that use both MambaCache and KVCache.

Usage:
    python benchmark_mamba_batching.py [--model MODEL_PATH] [--max-tokens N]
"""

import argparse
import time
from typing import List, Tuple

import mlx.core as mx


def test_batch_with_cache(model_path: str, max_tokens: int = 50) -> dict:
    """
    Test batch generation with prompt caches on a hybrid model.

    Returns dict with timing and success info.
    """
    from mlx_lm import load, generate
    from mlx_lm.generate import batch_generate

    results = {
        "model": model_path,
        "max_tokens": max_tokens,
        "tests": {},
    }

    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")

    # Load model
    start = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    results["load_time"] = load_time

    # Check if model uses MambaCache
    cache = model.make_cache()
    cache_types = [type(c).__name__ for c in cache[:5]]  # First 5 layers
    print(f"Cache types (first 5 layers): {cache_types}")
    results["cache_types"] = cache_types

    has_mamba = any("Mamba" in t or "Arrays" in t for t in cache_types)
    if not has_mamba:
        print("WARNING: Model does not use MambaCache. Testing anyway...")

    # Test 1: Single generation (baseline)
    print(f"\n--- Test 1: Single Generation (baseline) ---")
    prompt = "Write a short poem about artificial intelligence:"

    start = time.time()
    output = generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )
    single_time = time.time() - start
    print(f"Time: {single_time:.2f}s")
    print(f"Output: {output[:100]}...")
    results["tests"]["single"] = {"time": single_time, "success": True}

    # Test 2: Batch generation WITHOUT prompt caches
    print(f"\n--- Test 2: Batch Generation (no cache) ---")
    prompts = [
        "Write a haiku about the ocean:",
        "Write a haiku about mountains:",
        "Write a haiku about the city:",
        "Write a haiku about the forest:",
    ]
    tokenized = [tokenizer.encode(p) for p in prompts]  # List of lists, not mx.arrays

    start = time.time()
    batch_result = batch_generate(
        model, tokenizer, tokenized, max_tokens=max_tokens
    )
    batch_time = time.time() - start
    print(f"Time for {len(prompts)} prompts: {batch_time:.2f}s ({batch_time/len(prompts):.2f}s per prompt)")
    results["tests"]["batch_no_cache"] = {
        "time": batch_time,
        "per_prompt": batch_time / len(prompts),
        "success": True,
    }

    # Test 3: Generate with return_prompt_caches (creates caches)
    print(f"\n--- Test 3: Batch Generation WITH cache creation ---")

    start = time.time()
    batch_result_with_cache = batch_generate(
        model, tokenizer, tokenized,
        max_tokens=max_tokens,
        return_prompt_caches=True,
    )
    cache_create_time = time.time() - start
    print(f"Time: {cache_create_time:.2f}s")

    if batch_result_with_cache.caches:
        print(f"Caches returned: {len(batch_result_with_cache.caches)}")
        results["tests"]["batch_with_cache_creation"] = {
            "time": cache_create_time,
            "caches_created": len(batch_result_with_cache.caches),
            "success": True,
        }
    else:
        print("WARNING: No caches returned!")
        results["tests"]["batch_with_cache_creation"] = {
            "time": cache_create_time,
            "success": False,
            "error": "No caches returned",
        }

    # Test 4: Batch generation WITH prompt_caches (reuse caches)
    # This is the key test - it uses _merge_caches which now supports MambaCache
    print(f"\n--- Test 4: Batch Generation REUSING caches ---")

    followup_prompts = [
        " Now make it darker:",
        " Now make it happier:",
        " Now make it mysterious:",
        " Now make it peaceful:",
    ]
    followup_tokenized = [tokenizer.encode(p) for p in followup_prompts]  # List of lists

    try:
        start = time.time()
        batch_result_reuse = batch_generate(
            model, tokenizer, followup_tokenized,
            max_tokens=max_tokens,
            prompt_caches=batch_result_with_cache.caches,
        )
        cache_reuse_time = time.time() - start

        print(f"Time: {cache_reuse_time:.2f}s")
        print(f"SUCCESS! Batch generation with MambaCache prompt caches works!")

        # Show sample output
        for i, text in enumerate(batch_result_reuse.texts[:2]):
            print(f"\n  Prompt {i+1} output: {text[:80]}...")

        results["tests"]["batch_reuse_cache"] = {
            "time": cache_reuse_time,
            "success": True,
        }

    except ValueError as e:
        print(f"FAILED: {e}")
        results["tests"]["batch_reuse_cache"] = {
            "success": False,
            "error": str(e),
        }
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
        results["tests"]["batch_reuse_cache"] = {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_success = all(t.get("success", False) for t in results["tests"].values())

    if all_success:
        print("ALL TESTS PASSED!")

        # Calculate speedup from cache reuse
        if "batch_reuse_cache" in results["tests"] and "batch_no_cache" in results["tests"]:
            speedup = results["tests"]["batch_no_cache"]["time"] / results["tests"]["batch_reuse_cache"]["time"]
            print(f"\nCache reuse speedup: {speedup:.2f}x")
            results["speedup"] = speedup
    else:
        print("SOME TESTS FAILED!")
        for name, test in results["tests"].items():
            if not test.get("success", False):
                print(f"  - {name}: {test.get('error', 'Unknown error')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MambaCache batching")
    parser.add_argument(
        "--model",
        type=str,
        default="lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    results = test_batch_with_cache(args.model, args.max_tokens)

    return 0 if all(t.get("success", False) for t in results["tests"].values()) else 1


if __name__ == "__main__":
    exit(main())
