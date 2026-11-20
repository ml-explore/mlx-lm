# Copyright © 2026 Apple Inc.

import functools
import importlib.util
from importlib.metadata import EntryPoint, entry_points
from typing import Literal

__all__ = ["Kind", "module_name", "normalize_name"]

Kind = Literal["models", "tool_parsers", "chat_templates"]


def normalize_name(name: str) -> str:
    """A name as mlx-lm looks it up: in lower case, with ``_`` for ``-``."""
    return name.lower().replace("-", "_")


def module_name(kind: Kind, name: str) -> str:
    """The built-in module for ``name``, else the plug-in's; ValueError if none."""
    name = normalize_name(name)
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a valid name in mlx_lm.{kind}.")
    builtin = f"mlx_lm.{kind}.{name}"
    # Built-ins never read other packages' metadata.
    if importlib.util.find_spec(builtin) is not None:
        return builtin
    found = _entry_points(kind).get(name, [])
    if not found:
        raise ValueError(f"No {builtin} and no plug-in gives {name} in mlx_lm.{kind}.")
    return _plugin_module(f"mlx_lm.{kind}", found)


def _plugin_module(group: str, found: list[EntryPoint]) -> str:
    """The module of the one valid entry point for a name; ValueError if invalid."""
    ep = found[0]
    if len(found) > 1:
        first, second = found[0].dist.name, found[1].dist.name
        raise ValueError(
            f"{len(found)} packages give {ep.name} in {group}, for example "
            f"{first} and {second}."
        )
    if ep.attr is not None:
        raise ValueError(
            f"The entry point {ep.name} in {group} of {ep.dist.name} must be a "
            f"module, not {ep.value}."
        )
    if not _exists(ep.module):
        raise ValueError(
            f"{ep.dist.name} gives {ep.name} in {group}, but {ep.module} does not exist."
        )
    return ep.module


def _exists(module: str) -> bool:
    """Whether a module exists. Imports its parent packages."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError as e:
        # Only a missing parent means the module does not exist; any other
        # missing module is an import error of the plug-in itself.
        if e.name is not None and (module + ".").startswith(e.name + "."):
            return False
        raise


@functools.cache
def _entry_points(kind: Kind) -> dict[str, list[EntryPoint]]:
    """The entry points of the ``mlx_lm.{kind}`` group, by name."""
    found = {}
    for ep in entry_points(group=f"mlx_lm.{kind}"):
        found.setdefault(normalize_name(ep.name), []).append(ep)
    return found
