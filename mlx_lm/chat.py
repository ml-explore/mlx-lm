# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx

from .generate import stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import does_model_support_prompt_cache, load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = None
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_STEPS = 32
DEFAULT_THRESHOLD = 0.95


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
        "--block-length",
        type=int,
        default=DEFAULT_BLOCK_LENGTH,
        help="[Diffusion models only] Number of tokens per block",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="[Diffusion models only] Number of denoising iterations per block",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="[Diffusion models only] Confidence threshold for token acceptance",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={
            "trust_remote_code": True if args.trust_remote_code else None
        },
    )

    use_cache = does_model_support_prompt_cache(model)

    def print_help():
        print("The command list:")
        print("- 'q' to exit")
        print("- 'r' to reset the chat")
        print("- 'h' to display these commands")

    def reset_conversation():
        """Reset conversation history and prompt cache."""
        cache = make_prompt_cache(model, args.max_kv_size) if use_cache else None
        msgs = []
        if args.system_prompt is not None:
            msgs.append({"role": "system", "content": args.system_prompt})
        return cache, msgs

    print(f"[INFO] Starting chat session with {args.model}.")
    print_help()
    prompt_cache, messages = reset_conversation()

    while True:
        query = input(">> ")
        if query == "q":
            break
        if query == "r":
            prompt_cache, messages = reset_conversation()
            continue
        if query == "h":
            print_help()
            continue

        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        gen_kwargs = {
            "max_tokens": args.max_tokens,
            "sampler": make_sampler(
                args.temp,
                args.top_p,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            "prompt_cache": prompt_cache,
            "block_length": args.block_length,
            "steps": args.steps,
            "threshold": args.threshold,
        }

        assistant_response = ""
        for response in stream_generate(model, tokenizer, prompt, **gen_kwargs):
            print(response.text, flush=True, end="")
            assistant_response += response.text
        print()

        messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
