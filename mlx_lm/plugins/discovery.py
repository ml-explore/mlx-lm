# Copyright © 2026 Apple Inc.

import importlib
import importlib.util
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Any

import mlx.nn as nn

__all__ = ["import_if_exists", "load_model_class", "load_tool_parser", "lookup_name"]


def load_model_class(name: str) -> type[nn.Module]:
    """The Model class for ``name``; ValueError if none."""
    name = lookup_name(name)
    # Built-ins come first and never read package metadata.
    module = import_if_exists(f"mlx_lm.models.{name}")
    if module is not None:
        return module.Model
    return _entry_point("mlx_lm.models", name).load()


def load_tool_parser(name: str) -> Any:
    """The tool parser for ``name``; ValueError if none."""
    name = lookup_name(name)
    module = import_if_exists(f"mlx_lm.tool_parsers.{name}")
    if module is not None:
        return module
    return _entry_point("mlx_lm.tool_parsers", name).load()


def lookup_name(name: str) -> str:
    """``name`` in lower case, with ``_`` for ``-``."""
    return name.lower().replace("-", "_")


def import_if_exists(module: str) -> ModuleType | None:
    """The module, or None if there is none. Its parent package must exist."""
    spec = importlib.util.find_spec(module)
    # A folder without __init__.py has no origin.
    if spec is None or spec.origin is None:
        return None
    return importlib.import_module(module)


def _entry_point(group: str, name: str) -> EntryPoint:
    """The one entry point of ``group`` that gives ``name``; ValueError if none or more."""
    found = [ep for ep in entry_points(group=group) if lookup_name(ep.name) == name]
    if not found:
        raise ValueError(f"{name} is not built in, and no plug-in gives it in {group}.")
    if len(found) > 1:
        sources = ", ".join(sorted(f"{ep.name} from {ep.dist.name}" for ep in found))
        raise ValueError(
            f"{len(found)} entry points in {group} give {name}: {sources}."
        )
    return found[0]
