# Copyright © 2026 Apple Inc.

import dataclasses
import importlib
import importlib.util
import types
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from mlx_lm.models import MODEL_REMAPPING, lookup_name

__all__ = [
    "MODELS",
    "REGISTRIES",
    "TOOL_PARSERS",
    "Registry",
    "UnknownNameError",
    "import_if_exists",
    "list_plugins",
]


class UnknownNameError(ValueError):
    """No built-in or plug-in gives the name."""


# Hashed by identity, since the aliases are a dict.
@dataclasses.dataclass(frozen=True, eq=False)
class Registry:
    """The built-ins and plug-ins of one kind, such as models, by name.

    A built-in module gives its ``attribute``, or itself if that is None.
    """

    label: str
    _: dataclasses.KW_ONLY
    attribute: str | None = None
    aliases: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def load(self, name: str) -> Any:
        """The built-in for ``name``, else the one plug-in for it, else ValueError."""
        # Built-ins come first and never read package metadata.
        module = self.builtin_module(name)
        if module is not None:
            if self.attribute is None:
                return module
            return getattr(module, self.attribute)

        name = self._canonical_name(name)
        matches = [
            entry_point
            for entry_point in self._entry_points()
            if lookup_name(entry_point.name) == name
        ]

        if not matches:
            raise UnknownNameError(
                f"{name} is not built in, and no plug-in gives it in {self._package}."
            )
        if len(matches) > 1:
            sources = ", ".join(
                sorted(
                    f"{entry_point.name} from {entry_point.dist.name}"
                    for entry_point in matches
                )
            )
            raise ValueError(
                f"{len(matches)} entry points in {self._package} give {name}: "
                f"{sources}."
            )

        return matches[0].load()

    def builtin_module(self, name: str) -> types.ModuleType | None:
        """The module of the built-in for ``name``, or None."""
        module = import_if_exists(f"{self._package}.{self._canonical_name(name)}")
        if module is None:
            return None

        # Helper modules, such as mlx_lm.models.cache, have no Model.
        if self.attribute is not None and not hasattr(module, self.attribute):
            return None
        return module

    def plugins(self) -> list[dict[str, str | None]]:
        """The plug-ins as JSON records, with their errors. Imports their code."""
        return [
            {
                "project": entry_point.dist.name,
                "version": entry_point.dist.version,
                "kind": self.label,
                "name": entry_point.name,
                "value": entry_point.value,
                "error": self._load_error(entry_point),
            }
            for entry_point in self._entry_points()
        ]

    @property
    def _package(self) -> str:
        """The package of the built-ins, which is also the entry-point group."""
        return f"mlx_lm.{self.label}"

    def _canonical_name(self, name: str) -> str:
        """``name`` as the lookup uses it: normalized, then through the aliases."""
        name = lookup_name(name)
        return self.aliases.get(name, name)

    def _entry_points(self) -> metadata.EntryPoints:
        """The plug-in entry points."""
        return metadata.entry_points(group=self._package)

    def _load_error(self, entry_point: metadata.EntryPoint) -> str | None:
        """Why ``load`` does not give ``entry_point``, or None. Imports its code."""
        try:
            found = self.load(entry_point.name)
            given = entry_point.load()
        except Exception as exception:
            # Plug-in code can raise any error when it is imported.
            return f"{type(exception).__name__}: {exception}"

        # Only a built-in can win over the one entry point that gives a name.
        if found is not given:
            return f"shadowed by built-in {self._canonical_name(entry_point.name)}"
        return None


MODELS = Registry("models", attribute="Model", aliases=MODEL_REMAPPING)
TOOL_PARSERS = Registry("tool_parsers")
REGISTRIES = (MODELS, TOOL_PARSERS)


def list_plugins() -> list[dict[str, str | None]]:
    """The plug-ins of all registries, sorted by project. Imports their code."""
    plugins = []
    for registry in REGISTRIES:
        plugins.extend(registry.plugins())
    plugins.sort(key=lambda plugin: plugin["project"] or "")
    return plugins


def import_if_exists(module: str) -> types.ModuleType | None:
    """The module, or None if there is none. Its parent package must exist."""
    spec = importlib.util.find_spec(module)
    # A folder without __init__.py has no origin.
    if spec is None or spec.origin is None:
        return None

    return importlib.import_module(module)
