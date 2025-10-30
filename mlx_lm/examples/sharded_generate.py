# Copyright © 2025 Apple Inc.


import argparse

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM model parallel inference example")
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.3-70B-Instruct-4bit",
        help="HF repo or path to local model.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Write a quicksort in C++.",
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    model, tokenizer = sharded_load(args.model)

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens
    ):
        rprint(response.text, end="", flush=True)

    rprint()
    rprint("=" * 10)
    rprint(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    rprint(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    rprint(f"Peak memory: {response.peak_memory:.3f} GB")
