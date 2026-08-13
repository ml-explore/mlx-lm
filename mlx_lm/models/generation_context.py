# Copyright © 2026 Apple Inc.
"""Request-local logical-position context for batched generation forwards.

The generation package deliberately owns the public forward metadata while
models own the mechanics of consuming it.  Keeping the context in this small
module avoids a generate-to-model import cycle and makes the binding check
available to both the Qwen target and its native MTP head.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_ACTIVE_GENERATION_FORWARD = ContextVar("mlx_lm_generation_forward", default=None)


@contextmanager
def generation_forward_context(forward) -> Iterator[None]:
    """Expose one :class:`GenerationForward` to a compatible model call.

    The generation wrapper still controls acknowledgement activation.  This
    scope only transports immutable metadata; it is always reset on normal
    exit, cancellation, or a model error.
    """

    token = _ACTIVE_GENERATION_FORWARD.set(forward)
    try:
        yield
    finally:
        _ACTIVE_GENERATION_FORWARD.reset(token)


def consume_generation_positions(
    model: Any,
    cache: Any,
    input_tokens: Any,
    *,
    mtp: bool,
) -> Optional[Any]:
    """Return exact logical positions for the active Qwen call, if any.

    This is intentionally a fail-closed bridge.  A context for another model,
    cache, phase, or input shape cannot be consumed by this forward.  The
    acknowledgement object performs the final immutable value check while the
    model call is active.
    """

    forward = _ACTIVE_GENERATION_FORWARD.get()
    if forward is None or forward.logical_positions is None:
        return None
    if forward.model is not model:
        raise RuntimeError("generation_logical_position_model_mismatch")
    if forward.cache is not cache:
        raise RuntimeError("generation_logical_position_cache_mismatch")
    if tuple(forward.input_tokens.shape) != tuple(input_tokens.shape):
        raise RuntimeError("generation_logical_position_input_shape_mismatch")
    phase_value = getattr(forward.phase, "value", forward.phase)
    if (mtp and phase_value != "mtp_draft") or (not mtp and phase_value == "mtp_draft"):
        raise RuntimeError("generation_logical_position_phase_mismatch")
    acknowledgement = forward.logical_position_ack
    if acknowledgement is None:
        raise RuntimeError("generation_logical_position_ack_missing")
    acknowledgement._assert_consumer_binding(
        model=model,
        cache=cache,
        phase=forward.phase,
        input_shape=tuple(input_tokens.shape),
    )
    acknowledgement.acknowledge(forward.logical_positions)
    return forward.logical_positions
