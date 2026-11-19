# Copyright © 2026 Apple Inc.

"""Models, tool parsers and chat templates from other installed packages.

A package declares each one with an entry point. The name is the one mlx-lm
looks up: the ``model_type`` of a model, or the name of a tool parser or chat
template. The value is a module with the same contents as a built-in one:

    [project.entry-points."mlx_lm.models"]
    acme = "acme_lm.models.acme"

    [project.entry-points."mlx_lm.tool_parsers"]
    acme = "acme_lm.tool_parsers.acme"

    [project.entry-points."mlx_lm.chat_templates"]
    acme = "acme_lm.chat_templates.acme"

A built-in module always wins over a plug-in with the same name.
"""

import functools
import importlib.util
import warnings
from importlib.metadata import EntryPoint, entry_points

__all__ = ["module_name"]


def module_name(kind: str, name: str) -> str:
    """The module for ``name``: ``mlx_lm.{kind}.{name}``, else a plug-in's.

    Raises:
        ValueError: If neither exists, or if the plug-in entry points for
            ``name`` are not valid.
    """
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
    """The module of the one entry point that gives a name in a group.

    Raises:
        ValueError: If two packages give the name, or if the entry point names
            an object, not a module.
    """
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
def _entry_points(kind: str) -> dict[str, list[EntryPoint]]:
    """The entry points of the ``mlx_lm.{kind}`` group, by name."""
    found = {}
    for ep in entry_points(group=f"mlx_lm.{kind}"):
        found.setdefault(ep.name, []).append(ep)
    return found
