# Copyright © 2026 Apple Inc.

import dataclasses
import enum
import importlib
import importlib.util
import types
from collections.abc import Mapping
from importlib import metadata

from mlx_lm.models import MODEL_REMAPPING, lookup_name

__all__ = ["PluginKind", "import_if_exists"]


# Hashed by identity, since the aliases are a dict.
@dataclasses.dataclass(frozen=True, eq=False)
class _Settings:
    """The settings of a ``PluginKind``.

    A built-in gives its ``attribute`` (None: the module itself). ``aliases`` maps
    names to other built-ins.
    """

    label: str
    _: dataclasses.KW_ONLY
    attribute: str | None = None
    aliases: Mapping[str, str] = dataclasses.field(default_factory=dict)


class PluginKind(enum.Enum):
    """A kind of plug-in, and the lookup of its built-ins and entry points."""

    MODELS = _Settings("models", attribute="Model", aliases=MODEL_REMAPPING)
    TOOL_PARSERS = _Settings("tool_parsers")

    def __init__(self, settings: _Settings):
        self.label = settings.label
        self.attribute = settings.attribute
        self.aliases = settings.aliases

    @property
    def package(self) -> str:
        """The package of the built-ins, which is also the entry-point group."""
        return f"mlx_lm.{self.label}"

    def load(self, name: str) -> type | types.ModuleType:
        """The built-in for ``name``, else the plug-in; ValueError unless one."""
        # Built-ins come first and never read package metadata.
        found = self.builtin(name)
        if found is not None:
            return found

        name = self.resolve(name)
        matches = [
            entry_point
            for entry_point in self.entry_points()
            if lookup_name(entry_point.name) == name
        ]

        if not matches:
            raise ValueError(
                f"{name} is not built in, and no plug-in gives it in {self.package}."
            )
        if len(matches) > 1:
            sources = ", ".join(
                sorted(
                    f"{entry_point.name} from {entry_point.dist.name}"
                    for entry_point in matches
                )
            )
            raise ValueError(
                f"{len(matches)} entry points in {self.package} give {name}: "
                f"{sources}."
            )

        return matches[0].load()

    def error(self, entry_point: metadata.EntryPoint) -> str | None:
        """Why ``load`` does not give ``entry_point``, or None. Imports its code."""
        try:
            found = self.load(entry_point.name)
            given = entry_point.load()
        except Exception as exception:
            # Plug-in code can raise any error when it is imported.
            return f"{type(exception).__name__}: {exception}"

        # Only a built-in can win over the one entry that gives a name.
        if found is not given:
            return f"shadowed by built-in {self.resolve(entry_point.name)}"
        return None

    def builtin(self, name: str) -> type | types.ModuleType | None:
        """The built-in for ``name``, or None."""
        module = self.builtin_module(name)
        if module is None or self.attribute is None:
            return module
        return getattr(module, self.attribute)

    def builtin_module(self, name: str) -> types.ModuleType | None:
        """The module of the built-in for ``name``, or None."""
        module = import_if_exists(f"{self.package}.{self.resolve(name)}")
        if module is None:
            return None

        # Helper modules, such as mlx_lm.models.cache, have no Model.
        if self.attribute is not None and not hasattr(module, self.attribute):
            return None
        return module

    def resolve(self, name: str) -> str:
        """``name`` as the lookup uses it: normalized, then through the aliases."""
        name = lookup_name(name)
        return self.aliases.get(name, name)

    def entry_points(self) -> metadata.EntryPoints:
        """The plug-in entry points."""
        return metadata.entry_points(group=self.package)


def import_if_exists(module: str) -> types.ModuleType | None:
    """The module, or None if there is none. Its parent package must exist."""
    spec = importlib.util.find_spec(module)
    # A folder without __init__.py has no origin.
    if spec is None or spec.origin is None:
        return None

    return importlib.import_module(module)
