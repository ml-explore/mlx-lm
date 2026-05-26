# Copyright © 2024 Apple Inc.

"""Shared UI helpers for the mlx_lm command-line tools.

Centralizes the rich-based panel/progress/prompt rendering used by the chat
and training entry points and exposes an adaptive theme so the same markup
reads well on both light and dark terminal backgrounds.
"""

import os
import re
import shutil
import sys
import time

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.theme import Theme


def _osc11_to_rgb(timeout: float = 0.1):
    """Ask the terminal for its background color via OSC 11.

    Returns an (r, g, b) tuple in the 0-255 range, or None if the terminal
    does not respond (non-TTY, unsupported terminal, redirected stdio, ...).
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import select
        import termios
        import tty
    except ImportError:
        return None  # Windows / restricted environments

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None

    try:
        tty.setraw(fd)
        sys.stdout.write("\033]11;?\033\\")
        sys.stdout.flush()

        deadline = time.monotonic() + timeout
        buf = b""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not select.select([fd], [], [], remaining)[0]:
                break
            chunk = os.read(fd, 64)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\x07") or buf.endswith(b"\x1b\\"):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    match = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
    if not match:
        return None

    def _to_byte(hex_bytes: bytes) -> int:
        # OSC 11 components are typically 4 hex digits (16-bit) but some
        # terminals reply with 2. Normalize to 8 bits by scaling.
        digits = hex_bytes.decode("ascii")
        value = int(digits, 16)
        full = (1 << (4 * len(digits))) - 1
        return round(value * 255 / full) if full else 0

    return tuple(_to_byte(g) for g in match.groups())


def _detect_dark_background() -> bool:
    override = os.environ.get("MLX_LM_THEME", "").strip().lower()
    if override in ("dark", "light"):
        return override == "dark"

    rgb = _osc11_to_rgb()
    if rgb is not None:
        r, g, b = rgb
        # Perceived luminance (Rec. 601). < 128 ≈ dark background.
        return (0.299 * r + 0.587 * g + 0.114 * b) < 128

    # COLORFGBG is "fg;bg" or "fg;default;bg" with ANSI color indices.
    cfb = os.environ.get("COLORFGBG", "")
    if cfb:
        last = cfb.split(";")[-1].strip()
        try:
            bg = int(last)
            # 0-6 are the dim base colors and 8 is dark grey; 7 and 9-15
            # are the bright/light variants.
            return bg in (0, 1, 2, 3, 4, 5, 6, 8)
        except ValueError:
            pass

    # No signal from the terminal — assume dark, which is the modern default.
    return True


IS_DARK_BACKGROUND = _detect_dark_background()


def _make_theme() -> Theme:
    if IS_DARK_BACKGROUND:
        styles = {
            "ui.strong": "bold white",
            "ui.label": "grey70",
            "ui.muted": "grey62",
            "ui.heading": "bold grey62",
            "ui.dim": "grey50",
        }
    else:
        styles = {
            "ui.strong": "bold #000000",
            "ui.label": "#2a2a2a",
            "ui.muted": "grey42",
            "ui.heading": "bold #1a1a1a",
            "ui.dim": "grey62",
        }
    styles.update(
        {
            "ui.accent": "bold purple",
            "ui.border": "blue",
            "ui.good": "bold green",
            "ui.warn": "yellow",
            "progress.elapsed": "default",
            "progress.remaining": "default",
            "progress.percentage": "bold blue",
        }
    )
    return Theme(styles)


def make_console(**kwargs) -> Console:
    """Return a rich Console pre-loaded with the adaptive mlx_lm theme."""
    kwargs.setdefault("highlight", False)
    # Force truecolor so hex values in the theme survive instead of being
    # downgraded to ANSI colors that the terminal may remap.
    kwargs.setdefault("color_system", "truecolor")
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
        width = shutil.get_terminal_size((80, 24)).columns
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
        TextColumn("[ui.muted]·[/ui.muted]"),
        TimeElapsedColumn(),
        TextColumn("[ui.muted]<[/ui.muted]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=disable,
    )
