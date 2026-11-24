# Copyright © 2026 Apple Inc.

import argparse
import json
import sys
from importlib.metadata import EntryPoint, entry_points
from typing import Any

import mlx.nn as nn

from mlx_lm import models
from mlx_lm.plugins.discovery import import_if_exists, lookup_name

__all__ = ["main"]


def _model_problems(model: Any) -> list[str]:
    """The ways ``model`` breaks the model contract."""
    problems = []
    if not (isinstance(model, type) and issubclass(model, nn.Module)):
        problems.append("not an nn.Module subclass")
    if not hasattr(getattr(model, "model_args", None), "from_dict"):
        problems.append("no model_args with from_dict")
    return problems


def _tool_parser_problems(parser: Any) -> list[str]:
    """The ways ``parser`` breaks the tool parser contract."""
    problems = []
    for name in ("tool_call_start", "tool_call_end"):
        if not isinstance(getattr(parser, name, None), str):
            problems.append(f"{name} is not a str")
    if not callable(getattr(parser, "parse_tool_call", None)):
        problems.append("parse_tool_call is not callable")
    return problems


# The plug-in groups, and the contract check of each.
_CONTRACTS = {
    "mlx_lm.models": _model_problems,
    "mlx_lm.tool_parsers": _tool_parser_problems,
}


def _entries() -> list[EntryPoint]:
    """The entry points of all groups, sorted by project, group and name."""
    entries = [ep for group in _CONTRACTS for ep in entry_points(group=group)]
    return sorted(
        entries, key=lambda ep: (ep.dist.name, ep.group, lookup_name(ep.name))
    )


def _builtin_module(group: str, name: str) -> str | None:
    """The built-in module that wins over the entries that give ``name``, if any."""
    name = lookup_name(name)
    if group == "mlx_lm.models":
        # Aliases also go to built-ins.
        name = models.model_package({"model_type": name})
    # The built-ins of a group are in the package of the same name.
    module = f"{group}.{name}"
    if import_if_exists(module) is None:
        return None
    return module


def _lookup_problem(ep: EntryPoint, entries: list[EntryPoint]) -> str | None:
    """Why the lookup of the name of ``ep`` never gives ``ep``, or None."""
    module = _builtin_module(ep.group, ep.name)
    if module is not None:
        return f"shadowed by built-in {module}"

    name = lookup_name(ep.name)
    count = sum(
        other.group == ep.group and lookup_name(other.name) == name for other in entries
    )
    if count > 1:
        return f"{count} entries give {name}"
    return None


def _load_problems(ep: EntryPoint) -> list[str]:
    """The load error or the contract problems of ``ep``."""
    try:
        value = ep.load()
    except Exception as error:
        # Plug-in code can raise any error when it is imported.
        return [f"load fails: {type(error).__name__}: {error}"]
    return _CONTRACTS[ep.group](value)


def _describe(ep: EntryPoint) -> str:
    """``ep`` as ``[group] name = value (project version)``."""
    return f"[{ep.group}] {ep.name} = {ep.value} ({ep.dist.name} {ep.dist.version})"


def _kind(group: str) -> str:
    """The short name of an entry-point group, such as ``models``."""
    return group.removeprefix("mlx_lm.")


def _projects() -> list[dict[str, Any]]:
    """Each project: its name, version, and the names it gives of each kind."""
    projects = {}
    for ep in _entries():
        if ep.dist.name not in projects:
            projects[ep.dist.name] = {
                "name": ep.dist.name,
                "version": ep.dist.version,
                **{_kind(group): [] for group in _CONTRACTS},
            }
        projects[ep.dist.name][_kind(ep.group)].append(ep.name)
    return list(projects.values())


def _render(projects: list[dict[str, Any]]) -> str:
    """``projects`` as text: a line for each project, then one for each kind."""
    lines = []
    for project in projects:
        lines.append(f"{project['name']} {project['version']}")
        for kind in map(_kind, _CONTRACTS):
            if project[kind]:
                lines.append(f"  {kind}: {', '.join(project[kind])}")
    return "\n".join(lines)


def _check(names: list[str]) -> list[str]:
    """One line for each problem of the entries that give ``names``, or of all."""
    entries = _entries()
    given = {lookup_name(ep.name) for ep in entries}
    wanted = {lookup_name(name) for name in names} or given

    lines = [f"no entry gives {name}" for name in sorted(wanted - given)]
    for ep in entries:
        if lookup_name(ep.name) not in wanted:
            continue
        lookup = _lookup_problem(ep, entries)
        problems = [] if lookup is None else [lookup]
        problems.extend(_load_problems(ep))
        lines.extend(f"{_describe(ep)}: {problem}" for problem in problems)
    return lines


def main() -> None:
    """Run ``mlx_lm plugins list [--json]`` or ``mlx_lm plugins check [NAME ...]``."""
    parser = argparse.ArgumentParser(description="List and check the plug-ins.")
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser(
        "list", help="Show the entry points of the plug-ins."
    )
    list_parser.add_argument(
        "--json", action="store_true", help="Print the plug-ins as JSON."
    )
    check_parser = commands.add_parser(
        "check", help="Load the entry points and check their contracts."
    )
    check_parser.add_argument(
        "names", nargs="*", help="Check only the entries with these names."
    )
    args = parser.parse_args()

    if args.command == "list":
        projects = _projects()
        if args.json:
            print(json.dumps(projects, indent=2))
        else:
            print(_render(projects) or "No plug-ins found.")
        return
    problems = _check(args.names)
    print("\n".join(problems) or "No problems found.")
    if problems:
        sys.exit(1)
