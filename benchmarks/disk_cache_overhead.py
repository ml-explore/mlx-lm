# Copyright © 2026 Apple Inc.
"""Overhead benchmark for the disk-backed prompt cache.

Compares fetch_nearest_cache latency with disk=None vs
disk=DiskPromptCache(...) on repeat RAM-hit traffic. The disk-on path
should add a small overhead because the in-memory disk-trie lookup is
O(1) amortized and the mtime touch is async.

Usage:
    python -m benchmarks.disk_cache_overhead [--n 1000] [--cache-dir /tmp/bench-disk]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.disk_prompt_cache import DiskPromptCache
from mlx_lm.models.cache import KVCache, LRUPromptCache


def make_cache(num_layers: int = 4, ntokens: int = 128, dim: int = 64):
    cache = [KVCache() for _ in range(num_layers)]
    for c in cache:
        c.update_and_fetch(
            mx.random.normal((1, 1, ntokens, dim)),
            mx.random.normal((1, 1, ntokens, dim)),
        )
    mx.eval([c.state[0] for c in cache])
    return cache


def bench(use_disk: bool, n: int, cache_dir: Path) -> float:
    fake_model = cache_dir.parent / "model"
    fake_model.mkdir(exist_ok=True)
    (fake_model / "model.safetensors.index.json").write_text("{}")
    disk = None
    if use_disk:
        disk = DiskPromptCache(
            root=cache_dir,
            model_path=fake_model,
            max_bytes=10 * (1 << 30),
        )
        disk.start()
    try:
        ram = LRUPromptCache(max_size=10, disk=disk)
        tokens = list(range(128))
        kv = make_cache()
        model_id = disk.model_id if disk else "model"
        ram.insert_cache(model_id, tokens, kv, cache_type="user")
        if disk:
            disk._queue.join()

        # Warm-up
        for _ in range(20):
            ram.fetch_nearest_cache(model_id, tokens)

        t0 = time.perf_counter()
        for _ in range(n):
            ram.fetch_nearest_cache(model_id, tokens)
        t1 = time.perf_counter()
        return (t1 - t0) / n * 1e6  # μs per op
    finally:
        if disk is not None:
            disk.shutdown(timeout=5.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--cache-dir", type=Path, default=None)
    args = p.parse_args(argv)
    base_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="bench-disk-"))
    base_dir.mkdir(exist_ok=True)

    print(f"n={args.n} repeat fetches per run\n")

    runs_off = []
    for _ in range(3):
        runs_off.append(bench(False, args.n, base_dir / "off"))
    runs_on = []
    for _ in range(3):
        runs_on.append(bench(True, args.n, base_dir / "on"))

    median_off = statistics.median(runs_off)
    median_on = statistics.median(runs_on)
    overhead = (median_on - median_off) / median_off * 100

    print(
        f"disk=None:  median {median_off:.2f} μs/op  "
        f"(runs: {[f'{x:.2f}' for x in runs_off]})"
    )
    print(
        f"disk=on:    median {median_on:.2f} μs/op  "
        f"(runs: {[f'{x:.2f}' for x in runs_on]})"
    )
    print(f"overhead:   {overhead:+.1f}%")

    if overhead > 5.0:
        print("\nWARNING: overhead exceeds 5% threshold")
        return 1
    print("\nOK: overhead within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
