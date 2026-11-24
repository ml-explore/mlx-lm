# Copyright © 2026 Apple Inc.

import argparse
import collections
import json
import sys
from typing import NoReturn

from mlx_lm.plugins.discovery import list_plugins

__all__ = ["build_parser", "list_command", "main"]


def main(argv: list[str] | None = None) -> NoReturn:
    """Run ``mlx_lm plugin`` with ``argv`` (default: the command line)."""
    args = build_parser().parse_args(argv)

    if args.command == "list":
        status = list_command(as_json=args.json)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    sys.exit(status)


def build_parser() -> argparse.ArgumentParser:
    """The parser of ``mlx_lm plugin``."""
    parser = argparse.ArgumentParser(description="Commands for the plug-ins.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser(
        "list", help="Show the plug-ins of each project, and their errors."
    )
    list_parser.add_argument("--json", action="store_true", help="Print JSON.")
    return parser


def list_command(*, as_json: bool = False) -> int:
    """Print the plug-ins; return 1 if one has an error, else 0."""
    plugins = list_plugins()

    if as_json:
        message = json.dumps(plugins, indent=2)
    elif plugins:
        message = _render(plugins)
    else:
        message = "No plug-ins found."
    print(message)

    # A non-zero exit status lets scripts and CI see the errors.
    if all(plugin["error"] is None for plugin in plugins):
        return 0
    return 1


def _render(plugins: list[dict[str, str | None]]) -> str:
    """``plugins`` as text, under their project and kind."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for plugin in plugins:
        grouped[plugin["project"], plugin["version"]][plugin["kind"]].append(plugin)

    lines = []
    for (project, version), kinds in grouped.items():
        lines.append(f"{project} {version}")
        for kind, kind_plugins in kinds.items():
            lines.append(f"  {kind}")
            for plugin in kind_plugins:
                lines.append(f"    {plugin['name']} = {plugin['value']}")
                if plugin["error"] is not None:
                    lines.append(f"      {plugin['error']}")
    return "\n".join(lines)
