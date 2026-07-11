# Copyright © 2026 Apple Inc.

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.tuner.lora import LoRALinear

DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def comma_separated_ints(value):
    return tuple(int(item) for item in value.split(",") if item)


def configure_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark quantized LoRA precision, memory, and latency."
    )
    parser.add_argument("--input-dims", type=int, default=4096)
    parser.add_argument("--output-dims", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--ranks", type=comma_separated_ints, default=(8, 16, 32, 64))
    parser.add_argument("--bits", type=comma_separated_ints, default=(8, 6, 5, 4))
    parser.add_argument(
        "--adapter-counts",
        type=comma_separated_ints,
        default=(1, 8, 32, 64),
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rank-group-size", type=int, default=32)
    parser.add_argument("--base-bits", type=int, choices=(0, 4, 8), default=8)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser


def percentile(values, quantile):
    values = sorted(values)
    index = round((len(values) - 1) * quantile)
    return values[index]


def measure(function, warmup, iterations):
    for _ in range(warmup):
        mx.eval(function())
    mx.synchronize()

    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        mx.eval(function())
        mx.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "p90_ms": percentile(timings, 0.9),
    }


def adapter_bytes(module):
    return sum(
        value.nbytes
        for name, value in tree_flatten(module.parameters())
        if name.startswith("lora_")
    )


def compare(reference, candidate):
    reference = reference.astype(mx.float32)
    candidate = candidate.astype(mx.float32)
    difference = candidate - reference
    mx.eval(reference, candidate, difference)
    reference_norm = float(mx.linalg.norm(reference))
    difference_norm = float(mx.linalg.norm(difference))
    return {
        "relative_l2": difference_norm / (reference_norm + 1e-12),
        "cosine": float(
            mx.sum(reference * candidate)
            / (mx.linalg.norm(reference) * mx.linalg.norm(candidate) + 1e-12)
        ),
        "max_abs": float(mx.max(mx.abs(difference))),
    }


def benchmark_rank(args, rank, dtype):
    linear = nn.Linear(args.input_dims, args.output_dims, bias=False)
    linear.weight = linear.weight.astype(dtype)
    base = (
        nn.QuantizedLinear.from_linear(
            linear,
            group_size=args.group_size,
            bits=args.base_bits,
        )
        if args.base_bits
        else linear
    )
    lora = LoRALinear.from_base(base, r=rank, dropout=0.0, scale=1.0)
    lora.lora_a = (0.02 * mx.random.normal(lora.lora_a.shape)).astype(dtype)
    lora.lora_b = (0.02 * mx.random.normal(lora.lora_b.shape)).astype(dtype)
    x = mx.random.normal((args.tokens, args.input_dims)).astype(dtype)

    reference = (x @ lora.lora_a) @ lora.lora_b
    mx.eval(reference)
    fp_bytes = lora.lora_a.nbytes + lora.lora_b.nbytes
    fp_latency = measure(
        lambda: (x @ lora.lora_a) @ lora.lora_b,
        args.warmup,
        args.iterations,
    )
    fp_layer_latency = measure(
        lambda: lora(x),
        args.warmup,
        args.iterations,
    )
    result = {
        "rank": rank,
        "fp_adapter_bytes": fp_bytes,
        "fp_latency": fp_latency,
        "fp_layer_latency": fp_layer_latency,
        "quantized": {},
    }

    for bits in args.bits:
        quantized = lora.to_quantized(
            group_size=args.group_size,
            bits=bits,
            rank_group_size=args.rank_group_size,
        )
        candidate = quantized.lora_delta(x)
        mx.eval(candidate)
        size = adapter_bytes(quantized)
        latency = measure(
            lambda quantized=quantized: quantized.lora_delta(x),
            args.warmup,
            args.iterations,
        )
        layer_latency = measure(
            lambda quantized=quantized: quantized(x),
            args.warmup,
            args.iterations,
        )
        result["quantized"][str(bits)] = {
            **compare(reference, candidate),
            "adapter_bytes": size,
            "memory_reduction": 1 - size / fp_bytes,
            "latency": latency,
            "latency_ratio_vs_fp": latency["median_ms"] / fp_latency["median_ms"],
            "layer_latency": layer_latency,
            "layer_latency_ratio_vs_fp": (
                layer_latency["median_ms"] / fp_layer_latency["median_ms"]
            ),
            "multi_adapter_bytes": {
                str(count): count * size for count in args.adapter_counts
            },
        }
    return result


def print_table(results):
    header = (
        "rank bits size_KiB reduction rel_L2 cosine "
        "branch_ms branch_x layer_ms layer_x"
    )
    print(header)
    for result in results:
        rank = result["rank"]
        fp_size = result["fp_adapter_bytes"] / 1024
        fp_latency = result["fp_latency"]["median_ms"]
        fp_layer_latency = result["fp_layer_latency"]["median_ms"]
        print(
            f"{rank:>4} {'fp':>4} {fp_size:>8.1f} {'-':>9} {'-':>8} "
            f"{'-':>8} {fp_latency:>9.4f} {'1.00x':>8} "
            f"{fp_layer_latency:>8.4f} {'1.00x':>7}"
        )
        for bits, item in result["quantized"].items():
            print(
                f"{rank:>4} {bits:>4} {item['adapter_bytes'] / 1024:>8.1f} "
                f"{item['memory_reduction']:>8.1%} {item['relative_l2']:>8.4f} "
                f"{item['cosine']:>8.5f} "
                f"{item['latency']['median_ms']:>9.4f} "
                f"{item['latency_ratio_vs_fp']:>7.2f}x "
                f"{item['layer_latency']['median_ms']:>8.4f} "
                f"{item['layer_latency_ratio_vs_fp']:>6.2f}x"
            )


def main():
    args = configure_parser().parse_args()
    mx.random.seed(args.seed)
    dtype = DTYPES[args.dtype]
    results = [benchmark_rank(args, rank, dtype) for rank in args.ranks]
    payload = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mlx_version": mx.__version__,
        },
        "config": {
            "input_dims": args.input_dims,
            "output_dims": args.output_dims,
            "tokens": args.tokens,
            "ranks": args.ranks,
            "bits": args.bits,
            "adapter_counts": args.adapter_counts,
            "group_size": args.group_size,
            "rank_group_size": args.rank_group_size,
            "base_bits": args.base_bits,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "results": results,
    }
    print_table(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
