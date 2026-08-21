# Copyright © 2024 Apple Inc.

import re
import shutil
import sys
from contextlib import contextmanager
from functools import lru_cache

import mlx.core as mx
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, ProgressColumn, TextColumn
from rich.text import Text
from rich.theme import Theme

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def rprint(*args, **kwargs):
    """Print on rank 0 only; no-op on every other distributed worker."""
    if mx.distributed.init().rank() == 0:
        print(*args, **kwargs)


def _terminal_width(default: int = 120) -> int:
    return shutil.get_terminal_size(fallback=(default, 0)).columns or default


def _make_theme() -> Theme:
    return Theme(
        {
            "ui.strong": "bold",
            "ui.label": "default",
            "ui.muted": "grey50",
            "ui.heading": "bold",
            "ui.dim": "grey50",
            "ui.accent": "bold magenta",
            "ui.border": "blue",
            "ui.good": "bold green",
            "ui.warn": "yellow",
            "progress.percentage": "bold blue",
        }
    )


@lru_cache(maxsize=1)
def make_console() -> Console:
    """Return the shared rich Console pre-loaded with the mlx_lm theme."""
    return Console(
        theme=_make_theme(),
        highlight=False,
        color_system="truecolor",
        width=_terminal_width(),
    )


def print_header_panel(
    console: Console, title: str, rows: list[tuple[str, str]]
) -> None:
    """Render the boxed header used by the chat and training entry points."""
    label_w = max(len(k) for k, _ in rows)
    body = "\n".join(
        f"  [ui.label]{k.ljust(label_w)}[/ui.label]   [ui.strong]{v}[/ui.strong]"
        for k, v in rows
    )
    console.print(
        Panel(
            body,
            title=f"[ui.accent]{title}[/ui.accent]",
            title_align="left",
            border_style="ui.border",
            box=ROUNDED,
            padding=(0, 2),
        )
    )


def print_chat_help(console: Console) -> None:
    console.print(
        "  [ui.label]commands[/ui.label]    "
        "[ui.strong]q[/ui.strong] [ui.muted]exit[/ui.muted]   "
        "[ui.strong]r[/ui.strong] [ui.muted]reset[/ui.muted]   "
        "[ui.strong]h[/ui.strong] [ui.muted]help[/ui.muted]"
    )


def print_lora_run_header(console: Console, args) -> None:
    """Render the lora-run startup panel from a parsed args namespace."""
    type_label = args.fine_tune_type
    if args.fine_tune_type in ("lora", "dora"):
        lora_rank = args.lora_parameters.get("rank", "?")
        type_label = (
            f"{args.fine_tune_type} · {args.num_layers} layers · rank {lora_rank}"
        )
    elif args.fine_tune_type == "full":
        type_label = f"full · {args.num_layers} layers"

    lr = args.learning_rate if isinstance(args.learning_rate, (int, float)) else None
    lr_str = f"{lr:.1e}" if lr is not None else "schedule"

    rows = [
        ("model", str(args.model)),
        ("type", type_label),
        ("dataset", str(args.data)),
        ("optimizer", f"{args.optimizer} · lr {lr_str}"),
        ("batch · iters", f"{args.batch_size} · {args.iters:,}"),
        ("max seq", f"{args.max_seq_length:,}"),
    ]
    print_header_panel(console, "mlx_lm.lora", rows)


def corridor_input(console: Console) -> str:

    width = console.width
    dashes = "─" * max(width - 1, 10)
    with console.capture() as cap:
        console.print(f"[ui.muted]{dashes}[/ui.muted]")
        console.print()
        console.print(f"[ui.muted]{dashes}[/ui.muted]")
    sys.stdout.write(cap.get())
    sys.stdout.write("\033[2A\r")  # cursor up two rows onto the blank middle line
    sys.stdout.flush()

    with console.capture() as cap2:
        console.print("[ui.accent]›[/ui.accent] ", end="")
    prompt = _ANSI_RE.sub(lambda m: f"\x01{m.group(0)}\x02", cap2.get())
    try:
        return input(prompt)
    finally:
        # Cursor sits on the bottom-rule row; advance past it.
        sys.stdout.write("\n")
        sys.stdout.flush()


class SquareBar(ProgressColumn):
    """Progress bar rendered with █/░ blocks plus eighth-block sub-precision."""

    _EIGHTHS = "▏▎▍▌▋▊▉"  # 1/8 .. 7/8

    def __init__(self, bar_width: int = 40, complete_style: str = "blue"):
        super().__init__()
        self.bar_width = bar_width
        self.complete_style = complete_style

    def render(self, task):
        if not task.total:
            return Text("░" * self.bar_width, style="ui.dim")
        pct = min(max(task.completed / task.total, 0.0), 1.0)
        total_eighths = int(pct * self.bar_width * 8)
        full = total_eighths // 8
        rem = total_eighths % 8
        text = Text()
        text.append("█" * full, style=self.complete_style)
        used = full
        if rem > 0 and full < self.bar_width:
            text.append(self._EIGHTHS[rem - 1], style=self.complete_style)
            used += 1
        text.append("░" * (self.bar_width - used), style="ui.dim")
        return text


def make_train_progress(console: Console, *, disable: bool = False) -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description:<5}[/bold blue]"),
        SquareBar(bar_width=30, complete_style="blue"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[ui.muted]·[/ui.muted]"),
        TextColumn(
            "[bold blue]{task.completed:>5,}[/bold blue]"
            "[ui.muted]/{task.total:,}[/ui.muted]"
        ),
        console=console,
        transient=False,
        disable=disable,
    )


class TrainUI:
    """Helper class for rendering training progress and metrics."""

    def __init__(self, total_iters: int, rank: int = 0):
        self._rank = rank
        self._console = make_console()
        self._progress = make_train_progress(self._console, disable=(rank != 0))
        self._task = self._progress.add_task("train", total=total_iters)
        self._prev_train_loss = None

    def __enter__(self):
        if self._rank == 0:
            self._console.print(
                "  [ui.heading]iter   train_loss     tok/s     tokens[/ui.heading]"
            )
        self._progress.__enter__()
        return self

    def __exit__(self, *exc):
        return self._progress.__exit__(*exc)

    def advance(self):
        self._progress.advance(self._task)

    def report_train(self, it, train_loss, tokens_sec, trained_tokens):
        if self._rank != 0:
            return
        if self._prev_train_loss is None or train_loss <= self._prev_train_loss:
            arrow, style = "▼", "green"
        else:
            arrow, style = "▲", "yellow"
        self._prev_train_loss = train_loss
        self._console.print(
            f"  [ui.muted]{it:>4}[/ui.muted]    "
            f"[bold {style}]{train_loss:>5.3f} {arrow}[/bold {style}]    "
            f"[ui.strong]{tokens_sec:>5,.0f}[/ui.strong]    "
            f"[ui.muted]{trained_tokens / 1000:>5.1f}k[/ui.muted]"
        )

    @contextmanager
    def val_task(self, total: int):
        task_id = (
            self._progress.add_task("val", total=total) if self._rank == 0 else None
        )

        def advance():
            if task_id is not None:
                self._progress.advance(task_id)

        try:
            yield advance
        finally:
            if task_id is not None:
                self._progress.remove_task(task_id)

    def report_val(self, it, val_loss, val_time):
        if self._rank != 0:
            return
        self._console.print(
            f"  [ui.muted]{it:>4}[/ui.muted]    "
            f"[ui.accent]val[/ui.accent] "
            f"[ui.strong]{val_loss:>5.3f}[/ui.strong]    "
            f"[ui.muted]{val_time:.2f}s[/ui.muted]"
        )

    def report_save(self, checkpoint):
        if self._rank != 0:
            return
        self._console.print(
            f"  [ui.good]save[/ui.good]  " f"[ui.muted]{checkpoint.name}[/ui.muted]"
        )


class ChatUI:
    """Helper class for rendering the chat UI and streaming responses."""

    def __init__(self, args, rank: int = 0):
        self._rank = rank
        self._args = args
        self._console = make_console()

    def __enter__(self):
        if self._rank == 0:
            rows = [("model", str(self._args.model))]
            if self._args.adapter_path:
                rows.append(("adapter", str(self._args.adapter_path)))
            rows.append(("max tokens", f"{self._args.max_tokens:,}"))
            if self._args.system_prompt:
                sp = self._args.system_prompt
                if len(sp) > 60:
                    sp = sp[:57] + "..."
                rows.append(("system", sp))
            print_header_panel(self._console, "mlx_lm.chat", rows)
            print_chat_help(self._console)
        return self

    def __exit__(self, *exc):
        return False

    def prompt(self) -> str:
        if self._rank == 0:
            return corridor_input(self._console)
        return input("")

    def say_bye(self):
        if self._rank == 0:
            self._console.print("[ui.muted]bye[/ui.muted]")

    def say_reset(self):
        if self._rank == 0:
            self._console.print(
                "  [ui.good]reset[/ui.good] [ui.muted]context cleared[/ui.muted]"
            )

    def say_help(self):
        if self._rank == 0:
            print_chat_help(self._console)

    def stream_token(self, text: str):
        rprint(text, flush=True, end="")

    def end_turn(self, response):
        rprint()  # newline after the streamed line
        if self._rank != 0 or response is None:
            return
        self._console.print(
            f"  [ui.muted]{response.generation_tokens} tokens · "
            f"{response.generation_tps:.1f} tok/s · "
            f"prompt {response.prompt_tps:.1f} tok/s · "
            f"peak {response.peak_memory:.2f} GB[/ui.muted]"
        )
