# Copyright © 2026 Apple Inc.

import importlib
import importlib.util
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Any

import mlx.nn as nn

__all__ = ["import_if_exists", "load_model_class", "load_tool_parser", "lookup_name"]


def load_model_class(name: str) -> type[nn.Module]:
    """The Model class for ``name``; ValueError if none or invalid."""
    name = lookup_name(name)
    # Built-ins come first and never read package metadata.
    module = import_if_exists(f"mlx_lm.models.{name}")
    if module is not None:
        return module.Model

    ep = _entry_point("mlx_lm.models", name)
    model = ep.load()

    if not (isinstance(model, type) and issubclass(model, nn.Module)):
        raise ValueError(
            f"{ep.value} from {ep.dist.name} is not an nn.Module subclass."
        )
    model_args = getattr(model, "model_args", None)
    if not hasattr(model_args, "from_dict"):
        raise ValueError(
            f"{ep.value} from {ep.dist.name} has no model_args with from_dict."
        )
    return model


def load_tool_parser(name: str) -> Any:
    """The tool parser for ``name``; ValueError if none or invalid."""
    name = lookup_name(name)
    module = import_if_exists(f"mlx_lm.tool_parsers.{name}")
    if module is not None:
        return module

    ep = _entry_point("mlx_lm.tool_parsers", name)
    parser = ep.load()

    start = getattr(parser, "tool_call_start", None)
    end = getattr(parser, "tool_call_end", None)
    is_valid_parser = (
        isinstance(start, str)
        and isinstance(end, str)
        and callable(getattr(parser, "parse_tool_call", None))
    )
    if not is_valid_parser:
        raise ValueError(
            f"{ep.value} from {ep.dist.name} is not a tool parser "
            "(tool_call_start, tool_call_end, parse_tool_call)."
        )
    return parser


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
