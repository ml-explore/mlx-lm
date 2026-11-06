# Copyright © 2026 Apple Inc.

import functools
import itertools
import logging
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from mlx_lm import tokenizer_utils
from mlx_lm.generate import TextState
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.engine import errors, protocol

__all__ = ["from_messages", "from_text", "reserved_kwargs"]

_logger = logging.getLogger(__name__)

# Some templates reject empty messages, e.g. Ministral 3.
_PLACEHOLDER_USER = {"role": "user", "content": "."}
_PLACEHOLDER_ASSISTANT = {"role": "assistant", "content": "."}


def reserved_kwargs(kwargs: Mapping[str, Any]) -> list[str]:
    """The names in ``kwargs`` that a client may not set, sorted."""
    # They replace the template or change the tokens.
    reserved = {
        "add_generation_prompt",
        "chat_template",
        "conversation",
        "max_length",
        "messages",
        "padding",
        "return_tensors",
        "tokenize",
        "tokenizer_kwargs",
        "tools",
        "truncation",
    }
    return sorted(reserved & kwargs.keys())


def from_text(
    tokenizer: tokenizer_utils.TokenizerWrapper,
    text: str,
    *,
    params: settings.GenerationParams,
) -> protocol.Request:
    """The request of a text completion prompt."""
    try:
        tokens = tokenizer.encode(text)
    except Exception as e:
        raise errors.InvalidRequestError(f"The prompt cannot be tokenized: {e}") from e
    # Without a BOS token, an empty text has no tokens.
    if not tokens:
        raise errors.InvalidRequestError("The prompt has no tokens.")

    return protocol.Request(tokens, params=params)


def from_messages(
    tokenizer: tokenizer_utils.TokenizerWrapper,
    messages: Sequence[dict[str, Any]],
    *,
    params: settings.GenerationParams,
    tools: list[dict[str, Any]] | None = None,
    template_kwargs: dict[str, Any] | None = None,
    parse_tool_calls: bool = True,
) -> protocol.Request:
    """The request of a chat whose messages have the form chat templates expect.

    Raises ``InvalidRequestError`` for a chat that the template cannot render.
    """
    if not tokenizer.has_chat_template:
        raise errors.InvalidRequestError("The model has no chat template.")
    if tools and parse_tool_calls and not tokenizer.has_tool_calling:
        raise errors.InvalidRequestError(
            "The model has no tool call parser; use --tool-parser."
        )

    render_kwargs = {}
    if template_kwargs is not None:
        render_kwargs.update(template_kwargs)
    render_kwargs["tools"] = tools or None
    render = functools.partial(tokenizer.apply_chat_template, **render_kwargs)

    try:
        tokens = render(messages, add_generation_prompt=True)
    except Exception as e:
        raise errors.InvalidRequestError(f"The chat template failed: {e}") from e

    try:
        checkpoints = _checkpoints(render, messages, tokens=tokens)
    except Exception as e:
        # A cache hint must not fail the request.
        _logger.warning("No prompt cache checkpoints: %s", e)
        checkpoints = ()

    return protocol.Request(
        tokens,
        params=params,
        checkpoints=checkpoints,
        text_state=_text_state(tokenizer, tokens),
        parse_tool_calls=parse_tool_calls,
    )


def _text_state(
    tokenizer: tokenizer_utils.TokenizerWrapper, tokens: Sequence[int]
) -> TextState:
    """The state of the text after the prompt: reasoning if the prompt opens it."""
    if tokenizer.response_initial_state:
        return cast(TextState, tokenizer.response_initial_state)
    if tokenizer.has_thinking and _ends_in_think_block(tokenizer, tokens):
        return "reasoning"
    return "normal"


def _ends_in_think_block(
    tokenizer: tokenizer_utils.TokenizerWrapper, tokens: Sequence[int]
) -> bool:
    """Whether the prompt ends in a reasoning block that is still open."""
    start = tokenizer.rfind_think_start(tokens)
    if start <= tokenizer.rfind_think_end(tokens):
        return False
    # A marker in the user text is not at the end.
    return tokenizer.decode(tokens[start:]).strip() == tokenizer.think_start


def _checkpoints(
    render: Callable[..., Sequence[int]],
    messages: Sequence[dict[str, Any]],
    *,
    tokens: Sequence[int],
) -> tuple[tuple[int, protocol.CheckpointKind], ...]:
    """Where the next requests stop sharing the prompt."""
    checkpoints = {}
    system_position = _system_checkpoint(render, messages, tokens=tokens)
    if system_position is not None:
        checkpoints[system_position] = protocol.CheckpointKind.SYSTEM

    for position in _next_turn_checkpoints(render, messages, tokens=tokens):
        if position > (system_position or 0):
            checkpoints.setdefault(position, protocol.CheckpointKind.USER)

    return tuple(checkpoints.items())


def _system_checkpoint(
    render: Callable[..., Sequence[int]],
    messages: Sequence[dict[str, Any]],
    *,
    tokens: Sequence[int],
) -> int | None:
    """Where a chat with the same system prompt stops sharing the prompt."""
    system_messages = list(
        itertools.takewhile(lambda m: m["role"] == "system", messages)
    )
    if not (0 < len(system_messages) < len(messages)):
        return None

    other_tokens = render(
        [*system_messages, _PLACEHOLDER_USER], add_generation_prompt=False
    )
    position = _checkpoint_position(other_tokens, tokens)
    return position if 0 < position < len(tokens) - 1 else None


def _next_turn_checkpoints(
    render: Callable[..., Sequence[int]],
    messages: Sequence[dict[str, Any]],
    *,
    tokens: Sequence[int],
) -> list[int]:
    """Where the next turns of the chat stop sharing the prompt."""
    if messages[-1]["role"] not in ("user", "tool"):
        return []

    next_turns = [[*messages, _PLACEHOLDER_ASSISTANT, _PLACEHOLDER_USER]]
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if messages[-1]["role"] == "tool" and assistant_indices:
        # The next agent step keeps the reasoning.
        last_step = messages[assistant_indices[-1] :]
        next_turns.append([*messages, *last_step])

    positions = []
    for turn in next_turns:
        turn_tokens = render(turn, add_generation_prompt=False)
        positions.append(_checkpoint_position(turn_tokens, tokens))
    return positions


def _common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    same = map(operator.eq, a, b)
    return sum(itertools.takewhile(bool, same))


def _checkpoint_position(other_tokens: Sequence[int], tokens: Sequence[int]) -> int:
    """The tokens shared with ``tokens``, but the last, which can merge."""
    return _common_prefix_length(other_tokens, tokens) - 1
