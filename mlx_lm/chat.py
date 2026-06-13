# Copyright © 2023-2024 Apple Inc.

import argparse
import json

import mlx.core as mx

from .cli_ui import ChatUI
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
        "--chat-template-config",
        help="Additional JSON config for apply_chat_template, e.g. '{\"enable_thinking\": false}'",
        default=None,
    )
    parser.add_argument(
        "--draft-type",
        choices=["none", "ngram-simple", "ngram-mod"],
        default="none",
        help="Draft strategy for speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Number of draft tokens to propose.",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=None,
        help="N-gram window size. Defaults to 3 for ngram-simple and 16 for ngram-mod.",
    )
    parser.add_argument(
        "--disable-adaptive-gate",
        action="store_true",
        help="Disable the adaptive speculative decoding gate.",
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

    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(
            args.model,
            pipeline_group,
            tensor_group,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={"trust_remote_code": args.trust_remote_code},
            trust_remote_code=args.trust_remote_code,
        )

    with ChatUI(args, rank=rank) as ui:
        prompt_cache = make_prompt_cache(model, args.max_kv_size)
        template_kwargs = json.loads(args.chat_template_config or "{}")
        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        while True:
            query = ui.prompt()
            if query == "q":
                ui.say_bye()
                break
            if query == "r":
                prompt_cache = make_prompt_cache(model, args.max_kv_size)
                messages = []
                if args.system_prompt is not None:
                    messages.append({"role": "system", "content": args.system_prompt})
                ui.say_reset()
                continue
            if query == "h":
                ui.say_help()
                continue
            messages.append({"role": "user", "content": query})
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                **template_kwargs,
            )
            generate_kwargs = {}
            if args.draft_type != "none":
                generate_kwargs["draft_type"] = args.draft_type
                generate_kwargs["num_draft_tokens"] = args.num_draft_tokens
                generate_kwargs["disable_adaptive_gate"] = args.disable_adaptive_gate
                if args.ngram_size is not None:
                    generate_kwargs["ngram_size"] = args.ngram_size
            response_text = []
            accepted = 0
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
                **generate_kwargs,
            ):
                ui.stream_token(response.text)
                response_text.append(response.text)
                accepted += 1 if response.from_draft else 0
                last_response = response
            ui.end_turn(last_response)
            if last_response is not None and rank == 0:
                generated = last_response.generation_tokens
                acceptance = 100 * accepted / generated if generated else 0.0
                print(
                    "[stats] "
                    f"prompt={last_response.prompt_tokens} tok "
                    f"generated={generated} tok "
                    f"tok/s={last_response.generation_tps:.2f} "
                    f"accepted={accepted}/{generated} ({acceptance:.1f}%) "
                    f"peak={last_response.peak_memory:.2f} GB"
                )
            messages.append({"role": "assistant", "content": "".join(response_text)})


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
