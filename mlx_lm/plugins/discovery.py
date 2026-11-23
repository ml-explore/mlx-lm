# Copyright © 2026 Apple Inc.

import importlib
import importlib.util
import re
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal

import mlx.nn as nn

__all__ = ["load_model", "load_tool_parser", "lookup_name"]

PluginKind = Literal["models", "model_args", "tool_parsers"]

# Where each kind finds its built-ins: a module, and the name to take from it.
_BUILTINS = {
    "models": ("mlx_lm.models.{}", "Model"),
    "model_args": ("mlx_lm.models.{}", "ModelArgs"),
    "tool_parsers": ("mlx_lm.tool_parsers.{}", None),
}

# A plug-in entry point names an object: `module:attr`, with optional extras.
_OBJECT_VALUE = re.compile(
    r"(?P<module>\w+(?:\.\w+)*)\s*:\s*(?P<attr>\w+(?:\.\w+)*)\s*(?:\[.*\])?"
)


def load_model(name: str) -> tuple[type[nn.Module], type]:
    """The Model and ModelArgs classes for ``name``; ValueError if none or invalid."""
    model, model_source = _find("models", name)
    args, args_source = _find("model_args", name)
    if model_source != args_source:
        raise ValueError(
            f"The Model and ModelArgs of {name} must come from one project, not "
            f"{model_source} and {args_source}."
        )
    if not (isinstance(model, type) and issubclass(model, nn.Module)):
        raise ValueError(f"{model_source}: Model of {name} is not an nn.Module.")
    if not callable(getattr(args, "from_dict", None)):
        raise ValueError(f"{model_source}: ModelArgs of {name} has no from_dict.")
    return model, args


def load_tool_parser(name: str) -> Any:
    """The tool parser for ``name``; ValueError if none or invalid."""
    parser, source = _find("tool_parsers", name)
    start = getattr(parser, "tool_call_start", None)
    end = getattr(parser, "tool_call_end", None)
    is_parser = (
        isinstance(start, str)
        and start
        and isinstance(end, str)
        and callable(getattr(parser, "parse_tool_call", None))
    )
    if not is_parser:
        raise ValueError(
            f"{source}: the tool parser {name} needs tool_call_start, tool_call_end "
            "and parse_tool_call."
        )
    return parser


def lookup_name(name: str) -> str:
    """``name`` in lower case, with ``_`` for ``-``."""
    return name.lower().replace("-", "_")


def _find(kind: PluginKind, name: str) -> tuple[Any, str]:
    """The object for ``name``, and the project that gives it ("mlx-lm" if built in)."""
    name = lookup_name(name)
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a valid identifier.")

    template, attr = _BUILTINS[kind]
    builtin = template.format(name)
    # Built-ins come first, so other packages' metadata cannot break them.
    # A folder without __init__.py has no origin.
    spec = importlib.util.find_spec(builtin)
    if spec is not None and spec.origin is not None:
        module = importlib.import_module(builtin)
        return (module if attr is None else getattr(module, attr)), "mlx-lm"

    group = f"mlx_lm.{kind}"
    found = [ep for ep in entry_points(group=group) if lookup_name(ep.name) == name]
    if not found:
        raise ValueError(f"No {builtin} and no plug-in gives {name} in {group}.")
    if len(found) > 1:
        sources = ", ".join(sorted(f"{ep.name} from {ep.dist.name}" for ep in found))
        raise ValueError(
            f"{len(found)} entry points in {group} give {name}: {sources}."
        )
    ep = found[0]
    return _load(ep), ep.dist.name


def _load(ep: EntryPoint) -> Any:
    """The object a plug-in entry point names; ValueError if invalid or missing."""
    match = _OBJECT_VALUE.fullmatch(ep.value)
    if match is None:
        raise ValueError(
            f"The entry point {ep.name} in {ep.group} of {ep.dist.name} must name "
            f"an object (module:attr), not {ep.value!r}."
        )
    module = match["module"]
    if not _exists(module):
        raise ValueError(
            f"{ep.dist.name} gives {ep.name} in {ep.group}, but {module} "
            "does not exist."
        )
    found = importlib.import_module(module)
    for part in match["attr"].split("."):
        if not hasattr(found, part):
            raise ValueError(
                f"{ep.dist.name} gives {ep.name} in {ep.group}, but {ep.value} "
                "does not exist."
            )
        found = getattr(found, part)
    return found


def _exists(module: str) -> bool:
    """Whether a module exists. Imports its parent packages."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError as e:
        # Only a missing parent means no module; others are the plug-in's errors.
        if e.name is not None and (module + ".").startswith(e.name + "."):
            return False
        raise
