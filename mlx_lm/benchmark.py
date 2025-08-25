# Copyright © 2025 Apple Inc.

import argparse

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.generate import DEFAULT_MODEL
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import (
    fetch_from_hub,
    get_model_path,
)


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM benchmarking script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--prompt-tokens",
        "-p",
        default=512,
        help="Length of prompt",
        type=int,
    )
    parser.add_argument(
        "--generation-tokens",
        "-g",
        default=1024,
        help="Length of completion",
        type=int,
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        default=5,
        help="Number of timing trials",
        type=int,
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    mx.random.seed(0)

    model_path = args.model or DEFAULT_MODEL

    model_path, _ = get_model_path(model_path, revision=None)
    model, config, _ = fetch_from_hub(model_path, trust_remote_code=True)
    tokenizer = load_tokenizer(
        model_path,
        eos_token_ids=[],  # Empty to avoid early stopping
        tokenizer_config_extra={"trust_remote_code": True},
    )

    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens
    prompt = mx.random.randint(0, config["vocab_size"], (prompt_tokens,))

    def _bench():
        for response in stream_generate(
            model, tokenizer, prompt, max_tokens=generation_tokens
        ):
            pass
        return response

    print("Running warmup..")
    _bench()

    report_keys = ["prompt_tps", "generation_tps", "peak_memory"]
    print(f"Timing with {prompt_tokens=} and {generation_tokens=}.")
    responses = []
    for i in range(args.num_trials):
        response = _bench()
        responses.append(response)
        results = [(k, getattr(response, k)) for k in report_keys]
        results = [f"{k}={v:.3f}" for k, v in results]
        print(f"Trial {i+1}:  " + ", ".join(results))

    def avg(k):
        vals = (getattr(response, k) for response in responses)
        return sum(vals) / args.num_trials

    results = [(k, avg(k)) for k in report_keys]
    results = [f"{k}={v:.3f}" for k, v in results]
    print(f"Averages: " + ", ".join(results))


if __name__ == "__main__":
    main()
