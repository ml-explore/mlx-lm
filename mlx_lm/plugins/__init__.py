# Copyright © 2026 Apple Inc.

from mlx_lm.plugins.discovery import (
    import_if_exists,
    load_model_class,
    load_tool_parser,
    lookup_name,
)

__all__ = [
    "import_if_exists",
    "load_model_class",
    "load_tool_parser",
    "lookup_name",
    "main",
]


def main() -> None:
    """Run the ``mlx_lm plugins`` command."""
    # Import late: the command imports mlx_lm.models, which imports this package.
    from mlx_lm.plugins import cli

    cli.main()
