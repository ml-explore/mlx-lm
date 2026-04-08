# Copyright © 2025 Apple Inc.

"""CLI for DFlash block diffusion speculative decoding."""

import argparse
import sys
import time

import mlx.core as mx

from .utils import load
from .generate_dflash_v2 import block_diffusion_generate_step
from .sample_utils import make_sampler


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="DFlash block diffusion speculative decoding"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target model path or Hugging Face repo.",
    )
    parser.add_argument(
        "--draft",
        type=str,
        required=True,
        help="DFlash draft model path or Hugging Face repo.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate."
    )
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--block-size", type=int, default=None, help="Override block size.")
    parser.add_argument(
        "--quantize-draft",
        action="store_true",
        help="Quantize the draft model to 4-bit for faster inference.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-iteration stats."
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    print(f"Loading target model: {args.target}", file=sys.stderr)
    target_model, tokenizer = load(args.target)
    print(f"Loading draft model: {args.draft}", file=sys.stderr)
    draft_model, _ = load(args.draft)

    if args.quantize_draft:
        import mlx.nn as nn
        nn.quantize(draft_model, group_size=64, bits=4)
        print("Quantized draft model to 4-bit", file=sys.stderr)

    if args.block_size is not None:
        draft_model.block_size = args.block_size

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")

    tokens = []
    draft_count = 0
    start_time = time.time()

    for token_id, logprobs, from_draft in block_diffusion_generate_step(
        prompt=prompt,
        model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
    ):
        tokens.append(token_id if isinstance(token_id, int) else token_id)
        if from_draft:
            draft_count += 1
        text = tokenizer.decode([tokens[-1]])
        print(text, end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n\n--- {len(tokens)} tokens in {elapsed:.1f}s = {len(tokens)/elapsed:.1f} tok/s ---", file=sys.stderr)
    if tokens:
        print(f"--- Draft: {draft_count}/{len(tokens)} ({100*draft_count/len(tokens):.1f}%) ---", file=sys.stderr)


if __name__ == "__main__":
    main()
