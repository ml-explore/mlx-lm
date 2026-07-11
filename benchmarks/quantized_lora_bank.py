# Copyright © 2026 Apple Inc.

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.lora_pack import QuantizedLoRAAdapterPack


def comma_separated_ints(value):
    return tuple(int(item) for item in value.split(",") if item)


def configure_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark resident memory and routing for a LoRA adapter bank."
    )
    parser.add_argument("--input-dims", type=int, default=4096)
    parser.add_argument("--output-dims", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--bits", type=comma_separated_ints, default=(8, 4))
    parser.add_argument(
        "--adapter-counts", type=comma_separated_ints, default=(10, 50, 100)
    )
    parser.add_argument("--mixed-rows", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rank-group-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser


def percentile(values, quantile):
    values = sorted(values)
    return values[round((len(values) - 1) * quantile)]


def measure(function, warmup, iterations):
    for index in range(warmup):
        mx.eval(function(index))
    mx.synchronize()
    timings = []
    for index in range(iterations):
        started = time.perf_counter()
        mx.eval(function(index))
        mx.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "p90_ms": percentile(timings, 0.9),
    }


def compare(reference, candidate):
    reference = reference.astype(mx.float32)
    candidate = candidate.astype(mx.float32)
    difference = candidate - reference
    mx.eval(reference, candidate, difference)
    return {
        "relative_l2": float(mx.linalg.norm(difference))
        / (float(mx.linalg.norm(reference)) + 1e-12),
        "cosine": float(
            mx.sum(reference * candidate)
            / (mx.linalg.norm(reference) * mx.linalg.norm(candidate) + 1e-12)
        ),
        "max_abs": float(mx.max(mx.abs(difference))),
    }


def fp_mixed_delta(x, lora_a, lora_b, indices):
    z = mx.gather_mm(
        mx.expand_dims(x, -2),
        lora_a,
        rhs_indices=indices,
        sorted_indices=False,
    )
    return mx.gather_mm(
        z,
        lora_b,
        rhs_indices=indices,
        sorted_indices=False,
    ).squeeze(-2)


def make_adapters(args):
    maximum = max(args.adapter_counts)
    base = nn.Linear(args.input_dims, args.output_dims, bias=False)
    template = LoRALinear.from_base(
        base,
        r=args.rank,
        dropout=0.0,
        scale=1.0,
    )
    float_weights = []
    quantized = {bits: [] for bits in args.bits}
    for _ in range(maximum):
        lora_a = (0.02 * mx.random.normal(template.lora_a.shape)).astype(mx.float16)
        lora_b = (0.02 * mx.random.normal(template.lora_b.shape)).astype(mx.float16)
        float_weights.append((lora_a, lora_b))
        template.lora_a = lora_a
        template.lora_b = lora_b
        for bits in args.bits:
            quantized[bits].append(
                template.to_quantized(
                    group_size=args.group_size,
                    bits=bits,
                    rank_group_size=args.rank_group_size,
                )
            )
    return float_weights, quantized


def benchmark_count(args, count, float_weights, quantized_layers):
    lora_a = mx.stack([weights[0] for weights in float_weights[:count]])
    lora_b = mx.stack([weights[1] for weights in float_weights[:count]])
    fp_bytes = lora_a.nbytes + lora_b.nbytes
    request_x = mx.random.normal((1, args.input_dims)).astype(mx.float16)
    mixed_x = mx.random.normal((args.mixed_rows, args.input_dims)).astype(mx.float16)
    indices = mx.sort(
        (mx.arange(args.mixed_rows, dtype=mx.int32) % count).astype(mx.int32)
    )

    fp_request = measure(
        lambda iteration: (request_x @ lora_a[iteration % count])
        @ lora_b[iteration % count],
        args.warmup,
        args.iterations,
    )
    fp_mixed = measure(
        lambda _: fp_mixed_delta(
            mixed_x,
            lora_a,
            lora_b,
            indices,
        ),
        args.warmup,
        args.iterations,
    )
    reference = fp_mixed_delta(
        mixed_x,
        lora_a,
        lora_b,
        indices,
    )
    mx.eval(reference)
    result = {
        "adapter_count": count,
        "fp16": {
            "adapter_bytes": fp_bytes,
            "request_switch_latency": fp_request,
            "mixed_batch_latency": fp_mixed,
        },
        "quantized": {},
    }

    for bits in args.bits:
        pack = QuantizedLoRAAdapterPack.from_layers(quantized_layers[bits][:count])
        candidate = pack.delta(mixed_x, indices)
        mx.eval(candidate)
        request_latency = measure(
            lambda iteration, pack=pack: pack.delta_for_adapter(
                request_x, iteration % count
            ),
            args.warmup,
            args.iterations,
        )
        mixed_latency = measure(
            lambda _, pack=pack: pack.delta(mixed_x, indices),
            args.warmup,
            args.iterations,
        )
        result["quantized"][str(bits)] = {
            **compare(reference, candidate),
            "adapter_bytes": pack.adapter_bytes,
            "memory_reduction": 1 - pack.adapter_bytes / fp_bytes,
            "request_switch_latency": request_latency,
            "request_latency_ratio_vs_fp": request_latency["median_ms"]
            / fp_request["median_ms"],
            "mixed_batch_latency": mixed_latency,
            "mixed_latency_ratio_vs_fp": mixed_latency["median_ms"]
            / fp_mixed["median_ms"],
        }
    return result


def print_table(results):
    print(
        "count format memory_MiB reduction request_ms request_x "
        "mixed_ms mixed_x rel_L2 cosine"
    )
    for result in results:
        count = result["adapter_count"]
        fp = result["fp16"]
        print(
            f"{count:>5} {'fp16':>6} {fp['adapter_bytes'] / 2**20:>10.2f} "
            f"{'-':>9} {fp['request_switch_latency']['median_ms']:>10.4f} "
            f"{'1.00x':>9} {fp['mixed_batch_latency']['median_ms']:>8.4f} "
            f"{'1.00x':>7} {'-':>8} {'-':>8}"
        )
        for bits, item in result["quantized"].items():
            print(
                f"{count:>5} {('q' + bits):>6} "
                f"{item['adapter_bytes'] / 2**20:>10.2f} "
                f"{item['memory_reduction']:>8.1%} "
                f"{item['request_switch_latency']['median_ms']:>10.4f} "
                f"{item['request_latency_ratio_vs_fp']:>8.2f}x "
                f"{item['mixed_batch_latency']['median_ms']:>8.4f} "
                f"{item['mixed_latency_ratio_vs_fp']:>6.2f}x "
                f"{item['relative_l2']:>8.4f} {item['cosine']:>8.5f}"
            )


def main():
    args = configure_parser().parse_args()
    mx.random.seed(args.seed)
    float_weights, quantized_layers = make_adapters(args)
    results = [
        benchmark_count(args, count, float_weights, quantized_layers)
        for count in args.adapter_counts
    ]
    payload = {
        "config": {
            "input_dims": args.input_dims,
            "output_dims": args.output_dims,
            "rank": args.rank,
            "bits": args.bits,
            "adapter_counts": args.adapter_counts,
            "mixed_rows": args.mixed_rows,
            "group_size": args.group_size,
            "rank_group_size": args.rank_group_size,
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
