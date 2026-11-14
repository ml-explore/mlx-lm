# Copyright © 2026 Apple Inc.

import dataclasses
from typing import Any

import pydantic

from mlx_lm import tokenizer_utils
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import http
from mlx_lm.server_v2.api.anthropic import schema
from mlx_lm.server_v2.chat import rendering

__all__ = [
    "check_generation",
    "check_request",
    "generation_params",
    "parallel_tool_calls",
    "parse_tool_calls",
    "template_kwargs",
    "template_messages",
    "template_tools",
]

# The request fields that GenerationParams takes as they are.
_PARAM_FIELDS = {"max_tokens", "temperature", "top_p", "top_k"}

# Claude Code starts the system prompt with this block. It changes between
# conversations, so it would defeat the prompt cache.
_BILLING_HEADER = "x-anthropic-billing-header:"


def check_request(body: schema.CountTokensRequest) -> None:
    """Raise a 400 for the prompt fields that the server does not take."""
    if reserved := rendering.reserved_kwargs(body.chat_template_kwargs or {}):
        raise http.APIError(400, f"chat_template_kwargs cannot set {reserved}.")


def check_generation(body: schema.MessagesRequest) -> None:
    """Raise a 400 for the generation fields that the server does not take."""
    if body.tool_choice is not None and body.tool_choice.type in ("any", "tool"):
        raise http.APIError(400, "tool_choice: only auto and none are supported.")

    if body.output_config is not None and body.output_config.format is not None:
        raise http.APIError(400, "output_config.format is not supported.")

    # The template starts a new turn: it cannot continue the last message.
    if body.messages[-1].role == "assistant":
        message = "A last assistant message (prefill) is not supported."
        raise http.APIError(400, message)


def generation_params(
    body: schema.MessagesRequest, *, defaults: settings.GenerationParams
) -> settings.GenerationParams:
    """The generation parameters of the request: its values over ``defaults``.

    Raises a 400 for a value the server does not take.
    """
    given = body.model_dump(include=_PARAM_FIELDS, exclude_none=True)
    if body.stop_sequences is not None:
        given["stop"] = tuple(body.stop_sequences)

    try:
        return dataclasses.replace(defaults, **given)
    except pydantic.ValidationError as e:
        names = {"stop": "stop_sequences"}
        raise http.APIError.from_validation_error(e, names=names) from e


def template_kwargs(
    body: schema.CountTokensRequest, *, tokenizer: tokenizer_utils.TokenizerWrapper
) -> dict[str, Any]:
    """The chat template arguments of the request; chat_template_kwargs wins.

    An effort that the template refuses is left out: the template default applies.
    """
    kwargs = {}
    if body.thinking is not None:
        # Adaptive is on: the model decides how much to think. The tokenizer gives
        # the switch the name that the template uses.
        kwargs["enable_thinking"] = body.thinking.type != "disabled"
    if body.output_config is not None and body.output_config.effort is not None:
        effort = body.output_config.effort
        if rendering.takes_reasoning_effort(tokenizer, effort):
            kwargs["reasoning_effort"] = effort
    if body.chat_template_kwargs is not None:
        kwargs.update(body.chat_template_kwargs)
    return kwargs


def template_messages(body: schema.CountTokensRequest) -> list[dict[str, Any]]:
    """The system prompt and the messages in the form chat templates expect.

    Raises a 400 for a message that the server does not take.
    """
    messages = []
    system = _system_text(body.system)
    if system:
        messages.append({"role": "system", "content": system})

    # Claude Code can send system messages after the newest message.
    newest = max(
        (i for i, m in enumerate(body.messages) if m.role != "system"), default=-1
    )
    for i, message in enumerate(body.messages):
        messages += _template_turn(message, newest=i == newest)
    return messages


def template_tools(body: schema.CountTokensRequest) -> list[dict[str, Any]] | None:
    """The tools in the form chat templates expect; None without tools."""
    if not body.tools:
        return None
    return [_template_tool(tool) for tool in body.tools]


def parse_tool_calls(body: schema.CountTokensRequest) -> bool:
    """Whether the engine parses the tool calls of the reply.

    With tool_choice "none", the template still shows the tools, but calls stay text.
    """
    if not body.tools:
        return False
    return body.tool_choice is None or body.tool_choice.type != "none"


def parallel_tool_calls(body: schema.MessagesRequest) -> bool:
    """Whether the reply can have more than one tool call."""
    choice = body.tool_choice
    return choice is None or choice.disable_parallel_tool_use is not True


def _system_text(system: list[schema.RequestTextBlock] | str | None) -> str:
    """The text of the system prompt, without Claude Code's billing header."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n".join(
        block.text for block in system if not block.text.startswith(_BILLING_HEADER)
    )


def _template_turn(
    message: schema.RequestMessage, *, newest: bool
) -> list[dict[str, Any]]:
    """The template messages of a message: each tool result is a tool message.

    Tool results come first in a message, so the rest of it comes after them.
    """
    if isinstance(message.content, str):
        return [{"role": message.role, "content": message.content}]

    tool_messages, texts, reasonings, tool_calls = [], [], [], []
    for block in message.content:
        match block:
            case schema.RequestTextBlock():
                texts.append(block.text)
            case schema.RequestMediaBlock():
                # Only the newest: a client keeps the block in its history.
                if newest:
                    raise http.APIError(400, "Images and documents are not supported.")
                texts.append(_media_note(block))
            case schema.RequestThinkingBlock():
                reasonings.append(block.thinking)
            case schema.RequestToolUseBlock():
                tool_calls.append(_template_tool_call(block))
            case schema.RequestToolResultBlock():
                tool_message = {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": _tool_result_text(block.content),
                }
                tool_messages.append(tool_message)

    # A message of only tool results has no message of its own.
    if tool_messages and not (texts or reasonings or tool_calls):
        return tool_messages

    result = {"role": message.role, "content": "\n".join(texts)}
    if reasonings:
        result["reasoning_content"] = "\n".join(reasonings)
    # Llama 3 templates fail on an empty list.
    if tool_calls:
        result["tool_calls"] = tool_calls
    return [*tool_messages, result]


def _tool_result_text(
    content: str | list[schema.RequestTextBlock | schema.RequestMediaBlock],
) -> str:
    """The text of a tool result; its media is a note, so the agent loop goes on."""
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        match block:
            case schema.RequestTextBlock():
                texts.append(block.text)
            case schema.RequestMediaBlock():
                texts.append(_media_note(block))
    return "\n".join(texts)


def _media_note(block: schema.RequestMediaBlock) -> str:
    """The text in place of an image or a document."""
    kind = block.type
    return f"[{kind.capitalize()} omitted: the server does not support {kind}s.]"


def _template_tool_call(block: schema.RequestToolUseBlock) -> dict[str, Any]:
    """A tool_use block as a tool call in the form chat templates expect."""
    # Templates read the arguments as a mapping.
    function = {"name": block.name, "arguments": block.input}
    return {"id": block.id, "type": "function", "function": function}


def _template_tool(tool: schema.RequestTool) -> dict[str, Any]:
    """A tool as an OpenAI function tool: templates and tool parsers take this form."""
    # In the order of OpenAI clients: the template shows the JSON.
    function = {"name": tool.name}
    if tool.description is not None:
        function["description"] = tool.description
    function["parameters"] = tool.input_schema
    return {"type": "function", "function": function}
