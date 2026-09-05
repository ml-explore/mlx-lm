"""Compare decode kernels; exclude cache writes, model execution and scheduling."""

import argparse
import json
import platform
import statistics
import time

import mlx.core as mx
from eco_paged_attention import _paged_attention, paged_attention


def measure(fn, iterations):
    for _ in range(5):
        mx.eval(fn())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        mx.eval(fn())
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    mx.random.seed(42)
    rows = []
    for gqa, kv_heads in ((6, 4), (8, 2)):
        for batch in (1, 4):
            for length in (2048, 8192):
                page_size = 64
                blocks = length // page_size
                lengths_list = [length - i * (length // 8) for i in range(batch)]
                tables = mx.arange(batch * blocks, dtype=mx.uint32).reshape(
                    batch, blocks
                )[:, ::-1]
                lengths = mx.array(lengths_list, dtype=mx.uint32)
                k = mx.random.normal((batch * blocks, kv_heads, page_size, 256)).astype(
                    mx.bfloat16
                )
                v = mx.random.normal(k.shape).astype(k.dtype)
                q = mx.random.normal((batch, kv_heads * gqa, 1, 256)).astype(k.dtype)
                mask = (
                    mx.arange(length)[None, None, None, :]
                    < lengths[:, None, None, None]
                )

                def gather(pages):
                    return (
                        pages[tables]
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(batch, kv_heads, length, 256)
                    )

                kd, vd = gather(k), gather(v)
                mx.eval(q, k, v, tables, lengths, mask, kd, vd)

                def dense():
                    return mx.fast.scaled_dot_product_attention(
                        q, kd, vd, scale=1 / 16, mask=mask
                    )

                def gathered():
                    return mx.fast.scaled_dot_product_attention(
                        q, gather(k), gather(v), scale=1 / 16, mask=mask
                    )

                def direct():
                    return _paged_attention(q, k, v, tables, lengths, scale=1 / 16)

                checked = paged_attention(q, k, v, tables, lengths, scale=1 / 16)
                error = mx.max(
                    mx.abs(checked.astype(mx.float32) - dense().astype(mx.float32))
                ).item()
                assert error < 0.004, error
                row = dict(
                    gqa=gqa,
                    kv_heads=kv_heads,
                    batch=batch,
                    lengths=lengths_list,
                    max_abs_error=error,
                    sdpa_ms=measure(dense, args.iterations),
                    gather_sdpa_ms=measure(gathered, args.iterations),
                    paged_ms=measure(direct, args.iterations),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.output, "w") as output:
        json.dump(
            dict(
                mlx=mx.__version__,
                device=mx.device_info(),
                os=platform.platform(),
                dtype="bfloat16",
                page_size=64,
                iterations=args.iterations,
                scope="Warm decode only; cache-owned metadata; excludes writes and serving",
                results=rows,
            ),
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
