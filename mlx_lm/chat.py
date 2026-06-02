# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx

from .cli_ui import (
    corridor_input,
    make_console,
    print_chat_help,
    print_header_panel,
    printf,
)
from .generate import stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, sharded_load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def _print_chat_header(args, console):
    rows = [("model", str(args.model))]
    if args.adapter_path:
        rows.append(("adapter", str(args.adapter_path)))
    rows.append(("max tokens", f"{args.max_tokens:,}"))
    if args.system_prompt:
        sp = args.system_prompt
        if len(sp) > 60:
            sp = sp[:57] + "..."
        rows.append(("system", sp))
    print_header_panel(console, "mlx_lm.chat", rows)


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    console = make_console()

    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(args.model, pipeline_group, tensor_group)
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    if rank == 0:
        _print_chat_header(args, console)
        print_chat_help(console)

    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        if rank == 0:
            query = corridor_input(console)
        else:
            query = input("")
        if query == "q":
            if rank == 0:
                console.print("[ui.muted]bye[/ui.muted]")
            break
        if query == "r":
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            if rank == 0:
                console.print(
                    "  [ui.good]reset[/ui.good] [ui.muted]context cleared[/ui.muted]"
                )
            continue
        if query == "h":
            if rank == 0:
                print_chat_help(console)
            continue
        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        last_response = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                args.temp,
                args.top_p,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            prompt_cache=prompt_cache,
        ):
            printf(response.text, flush=True, end="")
            last_response = response
        printf()
        if rank == 0 and last_response is not None:
            console.print(
                f"  [ui.muted]{last_response.generation_tokens} tokens · "
                f"{last_response.generation_tps:.1f} tok/s · "
                f"prompt {last_response.prompt_tps:.1f} tok/s · "
                f"peak {last_response.peak_memory:.2f} GB[/ui.muted]"
            )


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
