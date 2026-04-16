# Copyright © 2023-2024 Apple Inc.

import argparse
import os
import readline

import mlx.core as mx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

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
DEFAULT_RENDER_WINDOW_SIZE = 20
DEFAULT_REFRESH_RATE = 10
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your responses are rendered in a terminal with "
    "Markdown support. Feel free to use Markdown formatting when appropriate: "
    "**bold**, *italic*, `inline code`, code blocks with syntax highlighting "
    "(```language), bullet lists, numbered lists, and headers."
)


def broadcast(s: str, group: mx.distributed.Group, src: int = 0) -> str:
    """
    Broadcast a string from the source rank to all other ranks in the group.
    """
    if group.size() == 1:
        return s
    if group.rank() == src:
        data = mx.array(s.encode("utf-8"))
        mx.eval(mx.distributed.all_sum(data.size, group=group))
        mx.eval(mx.distributed.all_sum(data, group=group))
    else:
        size = mx.distributed.all_sum(0, group=group).item()
        data = mx.distributed.all_sum(mx.zeros(size, dtype=mx.uint8), group=group)
        mx.eval(data)
        s = bytes(data).decode("utf-8")
    return s


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
        default=DEFAULT_XTC_THRESHOLD,
        help="Threshold the probs of each next token candidate to be sampled by XTC",
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
        help="System prompt to be used for the chat template "
        "(replaces the default Markdown-aware prompt)",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Disable the default system prompt entirely",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_RENDER_WINDOW_SIZE,
        help="The current size of the loading section",
    )
    parser.add_argument(
        "--refresh-rate",
        type=int,
        default=DEFAULT_REFRESH_RATE,
        help="The current refresh rate during the generation",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    console = (
        Console(
            force_terminal=True,
            highlight=True,
        )
        if rank == 0
        else None
    )

    def cprint(*pargs, **kwargs):
        """Print only from rank 0, using rich if available."""
        if rank == 0:
            console.print(*pargs, **kwargs)

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

    def print_help():
        """Display available commands."""
        if rank == 0:
            help_text = Text()
            help_text.append("Commands\n", style="bold underline")
            help_text.append("  q", style="bold cyan")
            help_text.append("  Exit the chat\n")
            help_text.append("  r", style="bold cyan")
            help_text.append("  Reset conversation\n")
            help_text.append("  h", style="bold cyan")
            help_text.append("  Show this help")
            cprint(Panel(help_text, border_style="dim"))

    cprint(
        Panel(
            f"[bold]Model:[/bold] {args.model}",
            title="[bold green]MLX Chat[/bold green]",
            border_style="green",
        )
    )

    print_help()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)

    while True:
        query = ""
        if rank == 0:
            try:
                cprint()
                query = input(">> ")
            except EOFError:
                query = "q"
            except KeyboardInterrupt:
                cprint("\n[dim]Use 'q' to quit[/dim]")
                query = ""

        query = broadcast(query, group, src=0)

        query = query.strip()
        if not query:
            continue

        if query == "q":
            cprint("[dim]Goodbye![/dim]")
            break

        if query == "r":
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            cprint("[green]✓ Conversation reset[/green]")
            continue

        if query == "h":
            print_help()
            continue

        messages = []
        if not args.no_system_prompt:
            if args.system_prompt:
                system_content = args.system_prompt
            else:
                system_content = DEFAULT_SYSTEM_PROMPT
            messages.append({"role": "system", "content": system_content})
        elif args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        sampler = make_sampler(
            args.temp,
            args.top_p,
            xtc_threshold=args.xtc_threshold,
            xtc_probability=args.xtc_probability,
            xtc_special_tokens=(tokenizer.encode("\n") + list(tokenizer.eos_token_ids)),
        )

        if rank == 0:
            response_text = ""
            finish_reason = None
            window_size = (
                args.window_size if args.window_size > 0 else DEFAULT_RENDER_WINDOW_SIZE
            )
            refresh_rate = (
                args.refresh_rate if args.refresh_rate > 0 else DEFAULT_REFRESH_RATE
            )

            with Live(
                Markdown(response_text),
                console=console,
                refresh_per_second=refresh_rate,
                transient=True,
            ) as live:
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    prompt_cache=prompt_cache,
                ):
                    response_text += response.text
                    finish_reason = response.finish_reason

                    lines = response_text.splitlines(keepends=True)
                    if len(lines) > window_size:
                        display_text = "".join(lines[-window_size:])
                    else:
                        display_text = response_text

                    live.update(
                        Panel(
                            Markdown(display_text),
                            title="[blue]Generating...[/blue]",
                            border_style="blue",
                        )
                    )
            cprint(Markdown(response_text))

            if finish_reason == "length":
                cprint(
                    f"\n[yellow]⚠️ Output truncated "
                    f"(max tokens: {args.max_tokens})[/yellow]"
                )
        else:
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
            ):
                pass


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
