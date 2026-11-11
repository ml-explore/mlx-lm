# Copyright © 2026 Apple Inc.

"""The API-agnostic messages that cross the engine boundary."""

import dataclasses
import enum
from typing import Self

from mlx_lm.generate import TextState, TextStateMachine
from mlx_lm.server_v2 import settings

__all__ = [
    "CheckpointKind",
    "Event",
    "FailedEvent",
    "FinishReason",
    "FinishedEvent",
    "Request",
    "TokenLogprob",
    "TokensEvent",
    "Update",
    "Usage",
]


class CheckpointKind(enum.StrEnum):
    """Where in the prompt a checkpoint is, as the cache store names it."""

    SYSTEM = "system"
    USER = "user"


@dataclasses.dataclass(frozen=True)
class Request:
    """What a caller asks the engine to generate from a prompt."""

    tokens: tuple[int, ...]
    _: dataclasses.KW_ONLY
    params: settings.GenerationParams = settings.GenerationParams()
    # (position, kind) pairs: store the prompt cache there during prefill.
    checkpoints: tuple[tuple[int, CheckpointKind], ...] = ()
    # Where a chat prompt ends, e.g. inside a reasoning block; None is plain text.
    text_state: TextState | None = None
    parse_tool_calls: bool = True

    def __post_init__(self):
        # The caller can change a list after the submit: keep an immutable copy.
        object.__setattr__(self, "tokens", tuple(self.tokens))


class FinishReason(enum.StrEnum):
    """Why a request ended."""

    EOS = "eos"
    STOP_STRING = "stop_string"
    LENGTH = "length"
    CANCELLED = "cancelled"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Usage:
    """The token counts of a request; the prompt tokens include the cached ones."""

    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenLogprob:
    """The log probability of a token, and the most likely tokens with theirs."""

    token: int
    logprob: float
    top: tuple[tuple[int, float], ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokensEvent:
    """New token ids, and the parsed text that is ready: it can be empty."""

    token_ids: tuple[int, ...]
    delta: TextStateMachine.Delta
    # One per token when the request asks for them, else empty.
    logprobs: tuple[TokenLogprob, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class FinishedEvent:
    """The end of a request that did not fail."""

    reason: FinishReason
    usage: Usage
    # Set with STOP_STRING.
    stop_string: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class FailedEvent:
    """The end of a request that failed; a caller raises its error."""

    error: Exception


# A request gets TokensEvents, then one FinishedEvent or FailedEvent, in step order.
Event = TokensEvent | FinishedEvent | FailedEvent


@dataclasses.dataclass(frozen=True)
class Update:
    """The new events of a request: its token events, and its end if it came."""

    events: list[TokensEvent]
    end: FinishedEvent | None = None

    @classmethod
    def from_events(cls, events: list[Event]) -> Self:
        """The update of a list of events. Raises the error of a failed request."""
        assert not any(isinstance(e, FinishedEvent | FailedEvent) for e in events[:-1])

        tokens = [event for event in events if isinstance(event, TokensEvent)]
        end = events[-1] if events else None
        if isinstance(end, FailedEvent):
            raise end.error
        return cls(tokens, end if isinstance(end, FinishedEvent) else None)
