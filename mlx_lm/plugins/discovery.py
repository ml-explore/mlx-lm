# Copyright © 2026 Apple Inc.

import functools
import importlib.util
from importlib.metadata import EntryPoint, entry_points
from typing import Literal, get_args

__all__ = ["PluginKind", "module_name", "lookup_name"]

PluginKind = Literal["models", "tool_parsers", "chat_templates"]


def module_name(kind: PluginKind, name: str) -> str:
    """The built-in module for ``name``, else the plug-in's; ValueError if none."""
    if kind not in get_args(PluginKind):
        raise ValueError(f"{kind!r} is not one of {', '.join(get_args(PluginKind))}.")
    name = lookup_name(name)
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a valid identifier.")

    builtin = f"mlx_lm.{kind}.{name}"
    # Built-ins never read other packages' metadata. Namespace folders have no origin.
    spec = importlib.util.find_spec(builtin)
    if spec is not None and spec.origin is not None:
        return builtin

    group = f"mlx_lm.{kind}"
    found = _entry_points(group).get(name, [])
    if not found:
        raise ValueError(f"No {builtin} and no plug-in gives {name} in {group}.")
    return _plugin_module(group, found)


def lookup_name(name: str) -> str:
    """``name`` in lower case, with ``_`` for ``-``."""
    return name.lower().replace("-", "_")


@functools.cache
def _entry_points(group: str) -> dict[str, list[EntryPoint]]:
    """The entry points of ``group``, by lookup name."""
    found = {}
    for ep in entry_points(group=group):
        found.setdefault(lookup_name(ep.name), []).append(ep)
    return found


def _plugin_module(group: str, found: list[EntryPoint]) -> str:
    """The module of the one valid entry point for a name; ValueError if invalid."""
    ep, *rest = found
    if rest:
        raise ValueError(
            f"{len(found)} projects give {ep.name} in {group}, for example "
            f"{ep.dist.name} and {rest[0].dist.name}."
        )
    match = ep.pattern.match(ep.value)
    if match is None or match["attr"] is not None:
        raise ValueError(
            f"The entry point {ep.name} in {group} of {ep.dist.name} must be a "
            f"module, not {ep.value!r}."
        )
    module = match["module"]
    if not _exists(module):
        raise ValueError(
            f"{ep.dist.name} gives {ep.name} in {group}, but {module} does not exist."
        )
    return module


def _exists(module: str) -> bool:
    """Whether a module exists. Imports its parent packages."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError as e:
        # Only a missing parent means no module; others are the plug-in's errors.
        if e.name is not None and (module + ".").startswith(e.name + "."):
            return False
        raise
