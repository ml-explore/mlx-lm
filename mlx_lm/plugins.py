# Copyright © 2026 Apple Inc.

import functools
import importlib.util
import warnings
from importlib.metadata import EntryPoint, entry_points
from typing import Literal

__all__ = ["Kind", "module_name"]

Kind = Literal["models", "tool_parsers", "chat_templates"]


def module_name(kind: Kind, name: str) -> str:
    """The built-in module for ``name``, else the plug-in's; ValueError if none."""
    builtin = f"mlx_lm.{kind}.{name}"
    found = _entry_points(kind).get(name, [])
    if importlib.util.find_spec(builtin) is not None:
        if found:
            warnings.warn(f"{builtin} hides the plug-in module {found[0].value}.")
        return builtin
    if not found:
        raise ValueError(f"No {builtin} and no plug-in gives {name} in mlx_lm.{kind}.")
    return _plugin_module(f"mlx_lm.{kind}", found)


def _plugin_module(group: str, found: list[EntryPoint]) -> str:
    """The module of the one valid entry point for a name; ValueError if invalid."""
    ep = found[0]
    if len(found) > 1:
        packages = " and ".join(e.dist.name for e in found)
        raise ValueError(f"Both {packages} give {ep.name} in {group}.")
    if ep.attr is not None:
        raise ValueError(
            f"The entry point {ep.name} in {group} of {ep.dist.name} must be a "
            f"module, not {ep.value}."
        )
    return ep.value


@functools.cache
def _entry_points(kind: Kind) -> dict[str, list[EntryPoint]]:
    """The entry points of the ``mlx_lm.{kind}`` group, by name."""
    found = {}
    for ep in entry_points(group=f"mlx_lm.{kind}"):
        found.setdefault(ep.name, []).append(ep)
    return found
