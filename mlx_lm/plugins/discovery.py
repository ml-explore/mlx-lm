# Copyright © 2026 Apple Inc.

import dataclasses
import enum
import functools
import importlib
import importlib.util
import types
from importlib import metadata
from typing import Any, Self

__all__ = [
    "Plugin",
    "PluginKind",
    "import_if_exists",
    "installed_plugins",
    "lookup_name",
]


@dataclasses.dataclass(frozen=True)
class _Settings:
    """The settings of a ``PluginKind``.

    A built-in gives its ``attribute`` (None: the module itself). ``aliases`` names
    the table in the kind's package that maps names to other built-ins.
    """

    label: str
    _: dataclasses.KW_ONLY
    attribute: str | None = None
    aliases: str | None = None


class PluginKind(enum.Enum):
    """A kind of plug-in, and the lookup of its built-ins and entry points."""

    MODELS = _Settings("models", attribute="Model", aliases="MODEL_REMAPPING")
    TOOL_PARSERS = _Settings("tool_parsers")

    def __init__(self, settings: _Settings):
        self.label = settings.label
        self.attribute = settings.attribute
        self.aliases = settings.aliases

    @property
    def package(self) -> str:
        """The package of the built-ins, which is also the entry-point group."""
        return f"mlx_lm.{self.label}"

    def load(self, name: str) -> Any:
        """The built-in for ``name``, else the plug-in; ValueError unless one."""
        # Built-ins come first and never read package metadata.
        found = self.builtin(name)
        if found is not None:
            return found

        name = self.resolve(name)
        matches = [
            entry_point
            for entry_point in self.entries()
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

    def builtin(self, name: str) -> Any:
        """The built-in for ``name``, or None."""
        module = import_if_exists(f"{self.package}.{self.resolve(name)}")
        if module is None or self.attribute is None:
            return module

        # Helper modules, such as mlx_lm.models.cache, have no Model.
        return getattr(module, self.attribute, None)

    def resolve(self, name: str) -> str:
        """``name`` as the lookup uses it: normalized, then through the aliases."""
        name = lookup_name(name)
        if self.aliases is None:
            return name

        aliases = getattr(importlib.import_module(self.package), self.aliases)
        return aliases.get(name, name)

    def entries(self) -> list[metadata.EntryPoint]:
        """The plug-in entry points."""
        return list(metadata.entry_points(group=self.package))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Plugin:
    """A plug-in of an installed project."""

    project: str
    version: str
    kind: PluginKind
    name: str
    value: str

    @classmethod
    def from_entry_point(
        cls, entry_point: metadata.EntryPoint, *, kind: PluginKind
    ) -> Self:
        """The plug-in that ``entry_point`` of ``kind`` describes."""
        return cls(
            project=entry_point.dist.name,
            version=entry_point.dist.version,
            kind=kind,
            name=entry_point.name,
            value=entry_point.value,
        )

    @functools.cached_property
    def error(self) -> str | None:
        """Why the lookup does not give this plug-in, or None. Imports its code."""
        entry_point = metadata.EntryPoint(
            name=self.name, value=self.value, group=self.kind.package
        )

        try:
            found = self.kind.load(self.name)
            given = entry_point.load()
        except Exception as exception:
            # Plug-in code can raise any error when it is imported.
            return f"{type(exception).__name__}: {exception}"

        # Only a built-in can win over the one entry that gives a name.
        if found is not given:
            return f"shadowed by built-in {self.kind.resolve(self.name)}"
        return None

    def to_json(self) -> dict[str, Any]:
        """The plug-in as JSON, with the kind as its label."""
        fields = {"kind": self.kind.label, "error": self.error}
        return dataclasses.asdict(self) | fields

    def render(self) -> str:
        """``name = value``, then the error, if any."""
        text = f"{self.name} = {self.value}"
        if self.error is not None:
            text += f"\n  {self.error}"
        return text


def installed_plugins(kind: PluginKind | None = None) -> list[Plugin]:
    """The plug-ins of ``kind``, or of all kinds, sorted by project name."""
    kinds = list(PluginKind) if kind is None else [kind]

    plugins = (
        Plugin.from_entry_point(entry_point, kind=plugin_kind)
        for plugin_kind in kinds
        for entry_point in plugin_kind.entries()
    )
    return sorted(plugins, key=lambda plugin: plugin.project)


def lookup_name(name: str) -> str:
    """``name`` in lower case, with ``_`` for ``-``."""
    return name.lower().replace("-", "_")


def import_if_exists(module: str) -> types.ModuleType | None:
    """The module, or None if there is none. Its parent package must exist."""
    spec = importlib.util.find_spec(module)
    # A folder without __init__.py has no origin.
    if spec is None or spec.origin is None:
        return None

    return importlib.import_module(module)
