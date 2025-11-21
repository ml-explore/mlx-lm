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
        "--prefill-chunk",
        type=int,
        default=1024,
        help="Maximum prompt tokens admitted per sequence during prefill.",
    )
    return parser


__all__ = ["create_arg_parser"]
