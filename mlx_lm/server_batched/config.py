# ABOUTME: Defines CLI arguments for continuous batching server features.
# ABOUTME: Exposes helper to extend server parser with batching options.

import argparse
from typing import Optional


def create_arg_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--enable-continuous-batching",
        action="store_true",
        help="Enable continuous batching scheduler for streaming requests.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Maximum concurrent sequences handled by the scheduler.",
    )
    parser.add_argument(
        "--max-tokens-per-step",
        type=int,
        default=4096,
        help="Token budget per scheduler iteration for prefill work.",
    )
    parser.add_argument(
        "--decode-unroll",
        type=int,
        default=4,
        help="Number of decode iterations to unroll per scheduler tick.",
    )
    parser.add_argument(
        "--decode-unroll-safe",
        dest="decode_unroll_safe",
        action="store_true",
        default=True,
        help="(default) Reuse paged decode views only when block-boundary safe.",
    )
    parser.add_argument(
        "--decode-unroll-unsafe",
        dest="decode_unroll_safe",
        action="store_false",
        help="Disable block-boundary safety guard when reusing paged decode views.",
    )
    parser.add_argument(
        "--decode-engine",
        choices=("dense", "paged", "paged-arrays", "paged-arrays+compile"),
        default="paged-arrays",
        help="Decode backend to use inside the continuous batching server.",
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=256,
        help="Maximum prompt tokens admitted per sequence during prefill.",
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
        default=None,
        help="Optional millisecond budget for the first paged prefill slice (<=0 disables adaptation).",
    )
    parser.add_argument(
        "--metal-profiling",
        action="store_true",
        help="Enable Metal command buffer profiling (macOS 16+/Xcode 16+).",
    )
    parser.add_argument(
        "--force-legacy-generator",
        action="store_true",
        help="Use legacy BatchGenerator instead of slot-indexed decode.",
    )
    parser.add_argument(
        "--attn-backend",
        choices=("auto", "dense", "paged"),
        default="auto",
        help="Select attention backend; auto enables paged when support is present.",
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=16,
        help="Block size used when allocating paged key/value storage.",
    )
    parser.add_argument(
        "--kv-pool-blocks",
        default="auto",
        help="Total paged KV blocks to reserve ('auto' derives from Metal working-set).",
    )
    parser.add_argument(
        "--kv-quant",
        choices=("none", "int4_v"),
        default="none",
        help="Optional KV quantization mode (int4 V-only currently supported).",
    )
    parser.add_argument(
        "--kv-quant-group-size",
        type=int,
        default=64,
        help="Group size used when computing quantized V scales.",
    )
    parser.add_argument(
        "--paged-vec-width",
        default="auto",
        help="Vector width override for paged attention kernels ('auto' keeps defaults).",
    )
    parser.add_argument(
        "--paged-threads-per-head",
        default="auto",
        help="Threads-per-head override for paged attention kernels ('auto' keeps defaults).",
    )
    return parser


__all__ = ["create_arg_parser"]
