# Copyright © 2026 Apple Inc.

__all__ = [
    "ContextLengthExceededError",
    "EngineError",
    "InvalidRequestError",
    "UnavailableError",
]


class InvalidRequestError(ValueError):
    """The request is bad: the caller can change it and try again."""


class ContextLengthExceededError(InvalidRequestError):
    """The prompt and max_tokens do not fit in the context length."""


class EngineError(RuntimeError):
    """The engine failed to run a request. ``__cause__`` is the original error."""


class UnavailableError(RuntimeError):
    """The engine does not take requests in its current status."""

    def __str__(self) -> str:
        return f"The engine status is {self.args[0]}."
