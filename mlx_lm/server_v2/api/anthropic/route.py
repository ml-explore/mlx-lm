# Copyright © 2026 Apple Inc.

import dataclasses
from typing import Any

import fastapi
from fastapi import responses

from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import app, http
from mlx_lm.server_v2.api.anthropic import inputs, outputs, schema
from mlx_lm.server_v2.chat import rendering, tool_call
from mlx_lm.server_v2.engine import protocol

__all__ = ["AnthropicRoute", "router"]


class AnthropicRoute(http.ErrorRoute):
    """A route that answers its errors in the Anthropic format."""

    error_body = staticmethod(outputs.error_body)


router = fastapi.APIRouter(route_class=AnthropicRoute)


@router.post("/v1/messages")
async def messages(
    body: schema.MessagesRequest, req: fastapi.Request, state: app.ServerStateDep
) -> responses.Response:
    http.log_request(body)
    inputs.check_request(body)
    inputs.check_generation(body)

    params = inputs.generation_params(body, defaults=state.config.default_params)
    parse_tool_calls = inputs.parse_tool_calls(body)
    engine_request = await _build_request(
        body, req, state, params=params, parse_tool_calls=parse_tool_calls
    )
    # Claude Code asks for more tokens than the context has room for.
    context_length = state.engine.config.context_length
    if context_length is not None:
        num_tokens = len(engine_request.tokens)
        room = context_length - num_tokens
        if room < 1:
            raise outputs.context_length_error(num_tokens, context_length)
        if room < params.max_tokens:
            params = dataclasses.replace(params, max_tokens=room)
            engine_request = dataclasses.replace(engine_request, params=params)

    tools = inputs.template_tools(body) if parse_tool_calls else None
    parser = tool_call.ToolCallParser(
        state.tokenizer,
        tools=tools,
        parallel_tool_calls=inputs.parallel_tool_calls(body),
    )
    handle = state.engine.submit(engine_request)
    message = outputs.MessageOutput(
        state.config.model_id,
        parser=parser,
        engine_updates=state.engine.updates(handle),
        input_tokens=len(engine_request.tokens),
        limited_by_context=params.max_tokens < body.max_tokens,
    )

    keepalive = state.config.keepalive
    return await _answer(req, message, stream=body.stream is True, keepalive=keepalive)


@router.api_route("/api/hello", methods=["GET", "HEAD"])
async def hello() -> responses.Response:
    # Claude Code sends this at start to open the connection early.
    return responses.Response()


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    body: schema.CountTokensRequest, req: fastapi.Request, state: app.ServerStateDep
) -> dict[str, Any]:
    http.log_request(body)
    inputs.check_request(body)

    # The generation parameters and the tool call parsing do not change the prompt.
    engine_request = await _build_request(
        body, req, state, params=state.config.default_params, parse_tool_calls=False
    )
    count = schema.MessageTokensCount(input_tokens=len(engine_request.tokens))
    return count.model_dump()


async def _build_request(
    body: schema.CountTokensRequest,
    req: fastapi.Request,
    state: app.ServerState,
    *,
    params: settings.GenerationParams,
    parse_tool_calls: bool,
) -> protocol.Request:
    """The engine request of the body's prompt, built on the prompt thread."""
    messages = inputs.template_messages(body)
    tools = inputs.template_tools(body)
    kwargs = {
        **state.config.template_kwargs,
        **inputs.template_kwargs(body, tokenizer=state.tokenizer),
    }

    build_request = state.prompts.run(
        rendering.from_messages,
        state.tokenizer,
        messages,
        params=params,
        tools=tools,
        template_kwargs=kwargs,
        parse_tool_calls=parse_tool_calls,
    )
    return await http.while_connected(req, build_request)


async def _answer(
    req: fastapi.Request,
    output: outputs.MessageOutput,
    *,
    stream: bool,
    keepalive: float,
) -> responses.Response:
    """The output as server-sent events, or as one JSON body."""
    if stream:
        return http.event_stream(
            output.stream(), keepalive_event=outputs.PING_EVENT, keepalive=keepalive
        )
    body = await http.while_connected(req, output.body())
    return responses.JSONResponse(body)
