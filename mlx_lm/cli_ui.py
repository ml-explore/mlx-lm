# Copyright © 2024 Apple Inc.

"""Shared UI helpers for the mlx_lm command-line tools.

Centralizes the rich-based panel/progress/prompt rendering used by the chat
and training entry points. The theme is hardcoded for a light terminal
background.
"""

import os
import re
import shutil
import sys

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
)
from rich.text import Text
from rich.theme import Theme


def _terminal_width(default: int = 120) -> int:
    """Best-effort terminal width.

    Under launchers like ``mlx.launch`` the worker's stdout is a pipe, so
    Rich's auto-detection falls back to 80 columns. Honor an explicit
    ``MLX_LM_WIDTH`` override, then ``COLUMNS``, then a real TTY query, and
    finally a generous default that's nicer than 80 on modern terminals.
    """
    for var in ("MLX_LM_WIDTH", "COLUMNS"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    width = shutil.get_terminal_size(fallback=(0, 0)).columns
    return width if width > 0 else default


def _make_theme() -> Theme:
    return Theme(
        {
            "ui.strong": "bold #000000",
            "ui.label": "#2a2a2a",
            "ui.muted": "grey42",
            "ui.heading": "bold #1a1a1a",
            "ui.dim": "grey62",
            "ui.accent": "bold purple",
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


def make_corridor_prompt(console: Console):
    """Return a callable that draws the chat input corridor.

    The returned callable draws the top/bottom rules around the input line,
    repositions the cursor onto the middle line, and returns the styled
    "›" prompt string. Pass that string to ``input()`` so readline treats
    the marker as part of the prompt — otherwise backspace will erase it.
    """

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def _readline_safe(text: str) -> str:
        # Wrap escape sequences in \x01..\x02 so readline doesn't count
        # them when computing the prompt's visible width.
        return _ANSI_RE.sub(lambda m: f"\x01{m.group(0)}\x02", text)

    def _draw() -> str:
        width = console.width
        dashes = "─" * max(width - 1, 10)
        with console.capture() as cap:
            console.print(f"[ui.muted]{dashes}[/ui.muted]")
            console.print()
            console.print(f"[ui.muted]{dashes}[/ui.muted]")
        sys.stdout.write(cap.get())
        # Move the cursor up two rows back onto the blank middle line.
        sys.stdout.write("\033[2A\r")
        sys.stdout.flush()
        with console.capture() as cap2:
            console.print("[ui.accent]›[/ui.accent] ", end="")
        return _readline_safe(cap2.get())

    return _draw


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
