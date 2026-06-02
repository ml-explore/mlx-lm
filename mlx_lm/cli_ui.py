# Copyright © 2024 Apple Inc.

import re
import shutil
import sys

import mlx.core as mx
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, ProgressColumn, TextColumn
from rich.text import Text
from rich.theme import Theme

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def printf(*args, **kwargs):
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


def make_console(**kwargs) -> Console:
    """Return a rich Console pre-loaded with the mlx_lm theme."""
    kwargs.setdefault("highlight", False)
    kwargs.setdefault("color_system", "truecolor")
    kwargs.setdefault("width", _terminal_width())
    return Console(theme=_make_theme(), **kwargs)


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
        TextColumn("[bold blue]train[/bold blue]"),
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
