# Copyright © 2026 Apple Inc.

import importlib
import importlib.util
import re
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Any

import mlx.nn as nn

__all__ = ["import_if_exists", "load_model_class", "load_tool_parser", "lookup_name"]

# A plug-in entry point names an object: `module:name`, with optional extras.
_OBJECT_VALUE = re.compile(r"(?P<module>\w+(?:\.\w+)*)\s*:\s*(?P<attr>\w+)\s*(\[.*\])?")


def load_model_class(name: str) -> type[nn.Module]:
    """The Model class for ``name``; ValueError if none or invalid."""
    name = _checked_name(name)
    # Built-ins come first and never read package metadata.
    module = import_if_exists(f"mlx_lm.models.{name}")
    if module is not None:
        return module.Model

    ep = _entry_point("mlx_lm.models", name)
    model = _load(ep)
    if not (isinstance(model, type) and issubclass(model, nn.Module)):
        raise _entry_point_error(ep, f"{ep.value} is not an nn.Module subclass")
    if not callable(getattr(getattr(model, "model_args", None), "from_dict", None)):
        raise _entry_point_error(ep, f"{ep.value} has no model_args with from_dict")
    return model


def load_tool_parser(name: str) -> Any:
    """The tool parser for ``name``; ValueError if none or invalid."""
    name = _checked_name(name)
    # Built-ins come first and never read package metadata.
    module = import_if_exists(f"mlx_lm.tool_parsers.{name}")
    if module is not None:
        return module

    ep = _entry_point("mlx_lm.tool_parsers", name)
    parser = _load(ep)
    start = getattr(parser, "tool_call_start", None)
    end = getattr(parser, "tool_call_end", None)
    is_parser = (
        isinstance(start, str)
        and start != ""
        and isinstance(end, str)
        and callable(getattr(parser, "parse_tool_call", None))
    )
    if not is_parser:
        raise _entry_point_error(
            ep,
            f"{ep.value} is not a tool parser "
            "(tool_call_start, tool_call_end, parse_tool_call)",
        )
    return parser


def import_if_exists(module: str) -> ModuleType | None:
    """The module, or None if it or a parent is missing. Imports the parents.

    A folder without ``__init__.py`` counts as missing.
    """
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError as e:
        # Only a missing parent means no module; others are errors in the code.
        if e.name is not None and (module + ".").startswith(e.name + "."):
            return None
        raise
    if spec is None or spec.origin is None:
        return None
    return importlib.import_module(module)


def lookup_name(name: str) -> str:
    """``name`` in lower case, with ``_`` for ``-``."""
    return name.lower().replace("-", "_")


def _checked_name(name: str) -> str:
    """The lookup name of ``name``; ValueError if it is not an identifier."""
    name = lookup_name(name)
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a valid identifier.")
    return name


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


def _load(ep: EntryPoint) -> Any:
    """The object an entry point names; ValueError if the value is invalid or missing."""
    match = _OBJECT_VALUE.fullmatch(ep.value)
    if match is None:
        raise _entry_point_error(ep, f"{ep.value!r} is not module:name")
    module = import_if_exists(match["module"])
    if module is None:
        raise _entry_point_error(ep, f"{match['module']} does not exist")
    if not hasattr(module, match["attr"]):
        raise _entry_point_error(ep, f"{match['module']} has no {match['attr']}")
    return getattr(module, match["attr"])


def _entry_point_error(ep: EntryPoint, problem: str) -> ValueError:
    """An error about an entry point, which names its project."""
    return ValueError(f"{ep.dist.name} gives {ep.name} in {ep.group}, but {problem}.")
