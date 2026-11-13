# Copyright © 2026 Apple Inc.

try:
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401
    import uvicorn  # noqa: F401
except ImportError as e:
    raise ImportError('The server needs: pip install "mlx-lm[server]"') from e

# For the "mlx_lm server_v2" subcommand.
from mlx_lm.server_v2.cli import main

__all__ = ["main"]
