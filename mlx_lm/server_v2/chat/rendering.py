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

__all__ = ["from_messages", "from_text", "reserved_kwargs", "takes_reasoning_effort"]

_logger = logging.getLogger(__name__)

# Some templates reject empty messages, e.g. Ministral 3.
_PLACEHOLDER_USER = {"role": "user", "content": "."}
_PLACEHOLDER_ASSISTANT = {"role": "assistant", "content": "."}
_SYSTEM_MARKERS = ("mlx-lm first system marker", "mlx-lm later system marker")


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


# Clients choose the effort, so the cache has a limit.
@functools.lru_cache(maxsize=64)
def takes_reasoning_effort(
    tokenizer: tokenizer_utils.TokenizerWrapper, effort: str
) -> bool:
    """Whether the chat template takes ``effort`` as the reasoning effort.

    Some templates refuse an effort they do not know, e.g. Qwen3.8 takes only
    xhigh, medium and low.
    """
    try:
        tokenizer.apply_chat_template(
            [_PLACEHOLDER_USER],
            reasoning_effort=effort,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return False
    return True


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
    system_turns = any(message["role"] == "system" for message in messages[1:])
    if system_turns and not _shows_system_turns(tokenizer):
        messages = _system_as_user(messages)
    if not _takes_user_after_tool(tokenizer):
        messages = _user_into_tool(messages)

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


@functools.cache
def _shows_system_turns(tokenizer: tokenizer_utils.TokenizerWrapper) -> bool:
    """Whether the chat template shows a system message after the first message.

    Some templates refuse the message, some drop it without an error, and some
    show only the last one. The probe has the shape of a Claude Code chat: some
    templates refuse a system message only where it breaks the user/assistant order.
    """
    first, later = _SYSTEM_MARKERS
    probe = [
        {"role": "system", "content": first},
        _PLACEHOLDER_USER,
        {"role": "system", "content": later},
        _PLACEHOLDER_ASSISTANT,
        _PLACEHOLDER_USER,
    ]
    try:
        text = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return False
    return all(marker in text for marker in _SYSTEM_MARKERS)


@functools.cache
def _takes_user_after_tool(tokenizer: tokenizer_utils.TokenizerWrapper) -> bool:
    """Whether the chat template takes a user message right after a tool result."""
    # Mistral templates need 9-character ids, and gpt-oss a tool description.
    call = {
        "id": "a" * 9,
        "type": "function",
        "function": {"name": "f", "arguments": {}},
    }
    tool = {
        "type": "function",
        "function": {"name": "f", "description": ".", "parameters": {}},
    }
    probe = [
        _PLACEHOLDER_USER,
        {**_PLACEHOLDER_ASSISTANT, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "."},
        _PLACEHOLDER_USER,
    ]
    try:
        tokenizer.apply_chat_template(
            probe, tools=[tool], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return False
    return True


def _system_as_user(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages with each system message after the first as user text.

    Consecutive user messages become one, for templates that need the roles to
    alternate.
    """
    result = []
    for i, message in enumerate(messages):
        if i > 0 and message["role"] == "system":
            message = {**message, "role": "user"}
        if message["role"] == "user" and result and result[-1]["role"] == "user":
            content = f"{result[-1]['content']}\n\n{message['content']}"
            result[-1] = {**result[-1], "content": content}
        else:
            result.append(message)
    return result


def _user_into_tool(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages with each user message right after a tool result joined to it."""
    result = []
    for message in messages:
        if message["role"] == "user" and result and result[-1]["role"] == "tool":
            content = f"{result[-1]['content']}\n\n{message['content']}"
            result[-1] = {**result[-1], "content": content}
        else:
            result.append(message)
    return result


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
