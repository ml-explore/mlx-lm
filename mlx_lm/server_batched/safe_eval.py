# ABOUTME: Provides a resilient mx.eval wrapper for server threads.
# ABOUTME: Swallows benign primitive-free arrays while logging in debug mode.

from __future__ import annotations

import logging

import mlx.core as mx


def safe_eval(value) -> None:
    """Eval helper that tolerates already materialized arrays."""
    try:
        mx.eval(value)
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        message = str(exc).lower()
        if "without a primitive" in message:
            logging.debug(
                "safe_eval.skip value=%s reason=%s", type(value).__name__, exc
            )
            return
        raise


__all__ = ["safe_eval"]
