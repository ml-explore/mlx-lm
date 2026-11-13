# Copyright © 2026 Apple Inc.

from typing import Any

import fastapi
from fastapi import responses

from mlx_lm.server_v2.api import app, http
from mlx_lm.server_v2.api.openai import inputs, outputs, schema
from mlx_lm.server_v2.chat import rendering, tool_call

__all__ = ["OpenAIRoute", "router"]


class OpenAIRoute(http.ErrorRoute):
    """A route that answers its errors in the OpenAI format."""

    error_body = staticmethod(outputs.error_body)


router = fastapi.APIRouter(route_class=OpenAIRoute)


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(
    body: schema.ChatRequest, req: fastapi.Request, state: app.ServerStateDep
) -> responses.Response:
    http.log_request(body)
    inputs.check_model(body.model, served=state.config.model_id)
    inputs.check_chat(body)

    params = inputs.chat_params(body, defaults=state.config.default_params)
    tools = body.tools or None
    kwargs = {**state.config.template_kwargs, **inputs.template_kwargs(body)}
    # With tool_choice "none", the template still shows the tools, but calls stay text.
    parse_tool_calls = tools is not None and body.tool_choice != "none"
    messages = [inputs.template_message(m) for m in body.messages]

    build_request = state.prompts.run(
        rendering.from_messages,
        state.tokenizer,
        messages,
        params=params,
        tools=tools,
        template_kwargs=kwargs,
        parse_tool_calls=parse_tool_calls,
    )
    engine_request = await http.while_connected(req, build_request)

    parser = tool_call.ToolCallParser(
        state.tokenizer,
        tools=tools if parse_tool_calls else None,
        parallel_tool_calls=body.parallel_tool_calls is not False,
    )
    handle = state.engine.submit(engine_request)
    chat = outputs.ChatOutput(
        state.config.model_id,
        parser=parser,
        tokenizer=state.tokenizer,
        engine_updates=state.engine.updates(handle),
        include_usage=inputs.include_usage(body),
    )

    keepalive = state.config.keepalive
    return await _answer(req, chat, stream=body.stream is True, keepalive=keepalive)


@router.post("/v1/completions")
async def completions(
    body: schema.CompletionRequest,
    req: fastapi.Request,
    state: app.ServerStateDep,
) -> responses.Response:
    http.log_request(body)
    inputs.check_model(body.model, served=state.config.model_id)
    inputs.check_completion(body)

    params = inputs.completion_params(body, defaults=state.config.default_params)

    build_request = state.prompts.run(
        rendering.from_text, state.tokenizer, body.prompt, params=params
    )
    engine_request = await http.while_connected(req, build_request)

    handle = state.engine.submit(engine_request)
    completion = outputs.CompletionOutput(
        state.config.model_id,
        tokenizer=state.tokenizer,
        engine_updates=state.engine.updates(handle),
        include_usage=inputs.include_usage(body),
    )

    keepalive = state.config.keepalive
    return await _answer(
        req, completion, stream=body.stream is True, keepalive=keepalive
    )


@router.get("/v1/models")
async def models(state: app.ServerStateDep) -> dict[str, Any]:
    card = _model_card(state)
    return schema.ModelList(object="list", data=[card]).to_dict()


@router.get("/v1/models/{model_id:path}")
async def model(model_id: str, state: app.ServerStateDep) -> dict[str, Any]:
    inputs.check_model(model_id, served=state.config.model_id)
    return _model_card(state).to_dict()


async def _answer(
    req: fastapi.Request,
    output: outputs.ChatOutput | outputs.CompletionOutput,
    *,
    stream: bool,
    keepalive: float,
) -> responses.Response:
    """The output as server-sent events, or as one JSON body."""
    if stream:
        return http.event_stream(
            output.stream(),
            keepalive_event=outputs.KEEPALIVE_EVENT,
            keepalive=keepalive,
        )
    body = await http.while_connected(req, output.body())
    return responses.JSONResponse(body)


def _model_card(state: app.ServerState) -> schema.ModelCard:
    """The card of the served model."""
    # A model without a known context length has no max_model_len.
    fields = {}
    if (context_length := state.engine.config.context_length) is not None:
        fields["max_model_len"] = context_length

    return schema.ModelCard(
        id=state.config.model_id,
        object="model",
        created=state.created,
        owned_by="mlx-lm",
        **fields,
    )
