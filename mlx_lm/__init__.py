# Copyright © 2023-2024 Apple Inc.

import os

from ._version import __version__

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

__all__ = [
    "__version__",
    "convert",
    "batch_generate",
    "generate",
    "stream_generate",
    "load",
]

_LAZY = {
    "convert": ".convert",
    "batch_generate": ".generate",
    "generate": ".generate",
    "stream_generate": ".generate",
    "load": ".utils",
}


def __getattr__(name):
    # Imported on demand so `mlx_lm.models.<x>` does not pull in transformers,
    # jinja2 and the rest of the text-generation stack.
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
