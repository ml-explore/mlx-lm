# Copyright © 2026 Apple Inc.

import argparse
import collections
import dataclasses
import json
import sys
from typing import Any

from mlx_lm.plugins.discovery import Plugin, installed_plugins

__all__ = ["build_parser", "list_command", "main"]


def list_command(*, as_json: bool = False) -> int:
    """Print the plug-ins; return 1 if one has an error, else 0."""
    plugins = installed_plugins()

    if as_json:
        message = json.dumps([_to_json(plugin) for plugin in plugins], indent=2)
    elif plugins:
        message = _render(plugins)
    else:
        message = "No plug-ins found."
    print(message)

    # A non-zero exit status lets scripts and CI see the errors.
    if all(plugin.error is None for plugin in plugins):
        return 0
    return 1


def _to_json(plugin: Plugin) -> dict[str, Any]:
    """``plugin`` as JSON, with the kind as its label."""
    fields = {"kind": plugin.kind.label, "error": plugin.error}
    return dataclasses.asdict(plugin) | fields


def _render(plugins: list[Plugin]) -> str:
    """``plugins`` as text, under their project and kind."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for plugin in plugins:
        grouped[plugin.project, plugin.version][plugin.kind].append(plugin)

    lines = []
    for (name, version), kinds in grouped.items():
        lines.append(f"{name} {version}")
        for kind, kind_plugins in kinds.items():
            lines.append(f"  {kind.label}")
            for plugin in kind_plugins:
                lines.append(f"    {plugin.name} = {plugin.value}")
                if plugin.error is not None:
                    lines.append(f"      {plugin.error}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The parser of ``mlx_lm plugin``."""
    parser = argparse.ArgumentParser(description="Commands for the plug-ins.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser(
        "list", help="Show the plug-ins of each project, and their errors."
    )
    list_parser.add_argument("--json", action="store_true", help="Print JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run ``mlx_lm plugin`` with ``argv`` (default: the command line)."""
    args = build_parser().parse_args(argv)

    if args.command == "list":
        status = list_command(as_json=args.json)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    sys.exit(status)
