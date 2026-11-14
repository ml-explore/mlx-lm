# Copyright © 2026 Apple Inc.

import dataclasses
import json
from typing import Any

import pydantic

from mlx_lm import tokenizer_utils
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import http
from mlx_lm.server_v2.api.openai import schema
from mlx_lm.server_v2.chat import rendering

__all__ = [
    "chat_params",
    "check_chat",
    "check_completion",
    "check_model",
    "completion_params",
    "include_usage",
    "template_kwargs",
    "template_message",
]

# The request fields that GenerationParams takes as they are.
_PARAM_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "xtc_probability",
    "xtc_threshold",
    "repetition_penalty",
    "repetition_context_size",
    "presence_penalty",
    "presence_context_size",
    "frequency_penalty",
    "frequency_context_size",
}


def check_chat(body: schema.ChatRequest) -> None:
    """Raise a 400 for the fields that the server does not take."""
    if body.response_format not in (None, {"type": "text"}):
        raise _unsupported(
            "response_format", "Only the text response format is supported."
        )

    if body.top_logprobs and not body.logprobs:
        raise http.APIError(
            400, "top_logprobs needs logprobs set to true.", param="top_logprobs"
        )

    if body.tool_choice not in (None, "auto", "none"):
        raise _unsupported("tool_choice", "Only auto and none are supported.")
    if body.tools and any(t.get("type") != "function" for t in body.tools):
        raise _unsupported("tools", "Only function tools are supported.")

    kwargs = body.chat_template_kwargs or {}
    if reserved := rendering.reserved_kwargs(kwargs):
        message = f"chat_template_kwargs cannot set {reserved}."
        raise http.APIError(400, message, param="chat_template_kwargs")


def check_completion(body: schema.CompletionRequest) -> None:
    """Raise a 400 for the fields that the server does not take."""
    if not isinstance(body.prompt, str):
        raise _unsupported("prompt", "Only a string prompt is supported.")
    if body.echo:
        raise _unsupported("echo", "echo is not supported.")
    if body.suffix:
        raise _unsupported("suffix", "suffix is not supported.")
    if body.best_of not in (None, 1):
        raise _unsupported("best_of", "Only best_of=1 is supported.")


def check_model(model: str | None, *, served: str) -> None:
    """Raise a 404 for a model other than the one the server serves."""
    # A request without a model gets the served one, as in mlx_lm.server.
    if model is not None and model != served:
        message = f"The model {model} does not exist. The server is serving {served}."
        raise http.APIError(404, message, param="model", code="model_not_found")


def chat_params(
    body: schema.ChatRequest, *, defaults: settings.GenerationParams
) -> settings.GenerationParams:
    """The generation parameters of a chat request: its values over ``defaults``."""
    # max_completion_tokens replaces max_tokens.
    max_tokens = body.max_completion_tokens
    if max_tokens is None:
        max_tokens = body.max_tokens

    # The top count; without top_logprobs, the sampled token only.
    logprobs = None
    if body.logprobs:
        logprobs = 0 if body.top_logprobs is None else body.top_logprobs

    return _params(body, defaults=defaults, max_tokens=max_tokens, logprobs=logprobs)


def completion_params(
    body: schema.CompletionRequest, *, defaults: settings.GenerationParams
) -> settings.GenerationParams:
    """The generation parameters of a completion: its values over ``defaults``."""
    return _params(
        body, defaults=defaults, max_tokens=body.max_tokens, logprobs=body.logprobs
    )


def template_kwargs(
    body: schema.ChatRequest, *, tokenizer: tokenizer_utils.TokenizerWrapper
) -> dict[str, Any]:
    """The chat template arguments of the request; chat_template_kwargs wins.

    An effort that the template refuses is left out: the template default applies.
    """
    kwargs = {}
    effort = body.reasoning_effort
    if effort is not None and rendering.takes_reasoning_effort(tokenizer, effort):
        kwargs["reasoning_effort"] = effort
    if body.chat_template_kwargs is not None:
        kwargs.update(body.chat_template_kwargs)
    return kwargs


def template_message(message: schema.RequestMessage) -> dict[str, Any]:
    """A message in the form chat templates expect.

    Raises a 400 for a message that the server does not take.
    """
    result = message.model_dump(exclude={"content", "tool_calls"})

    # Most templates know only "system", its older name.
    if message.role == "developer":
        result["role"] = "system"

    if isinstance(message.content, list):
        if any(part.type != "text" for part in message.content):
            raise _unsupported("messages", "Only text content parts are supported.")
        result["content"] = "".join(part.text for part in message.content)
    else:
        result["content"] = "" if message.content is None else message.content

    if "reasoning" in result and "reasoning_content" not in result:
        result["reasoning_content"] = result.pop("reasoning")

    # Llama 3 templates fail on an empty list.
    if message.tool_calls:
        result["tool_calls"] = [_template_tool_call(tc) for tc in message.tool_calls]

    return result


def include_usage(body: schema.BaseRequest) -> bool:
    """Whether a stream ends with a chunk that has the token usage."""
    options = body.stream_options
    return options is not None and options.include_usage is True


def _params(
    body: schema.BaseRequest,
    *,
    defaults: settings.GenerationParams,
    max_tokens: int | None,
    logprobs: int | None,
) -> settings.GenerationParams:
    """The request's values over ``defaults``; ``logprobs`` is the top count.

    Raises a 400 for a value the server does not take.
    """
    if body.n not in (None, 1):
        raise _unsupported("n", "Only n=1 is supported.")

    given = body.model_dump(include=_PARAM_FIELDS, exclude_none=True)
    if given.get("top_k") == -1:
        given["top_k"] = 0
    if max_tokens is not None:
        given["max_tokens"] = max_tokens
    if logprobs is not None:
        given["logprobs"] = logprobs
    if body.stop is not None:
        given["stop"] = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop)

    if body.logit_bias:
        try:
            given["logit_bias"] = {int(k): v for k, v in body.logit_bias.items()}
        except ValueError as e:
            raise http.APIError(
                400, "logit_bias keys must be token ids.", param="logit_bias"
            ) from e

    try:
        params = dataclasses.replace(defaults, **given)
    except pydantic.ValidationError as e:
        raise http.APIError.from_validation_error(e) from e

    if body.seed is not None and params.temperature > 0:
        message = (
            "seed is supported only at temperature 0, but the temperature is "
            f"{params.temperature}. Set temperature to 0 in the request."
        )
        raise _unsupported("seed", message)

    return params


def _unsupported(param: str, message: str) -> http.APIError:
    return http.APIError(400, message, param=param, code="unsupported_value")


def _template_tool_call(tool_call: schema.RequestToolCall) -> dict[str, Any]:
    """A tool call of a message in the form chat templates expect."""
    arguments = tool_call.function.arguments
    # Templates read the arguments as a mapping.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, RecursionError) as e:
            raise http.APIError(
                400, f"Tool call arguments are not valid JSON: {e}"
            ) from e

    if not isinstance(arguments, dict):
        raise http.APIError(400, "Tool call arguments must be an object.")

    result = tool_call.model_dump()
    result["function"]["arguments"] = arguments
    return result
