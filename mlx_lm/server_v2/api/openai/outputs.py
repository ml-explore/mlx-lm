# Copyright © 2026 Apple Inc.

import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from mlx_lm import tokenizer_utils
from mlx_lm.server_v2.api import http
from mlx_lm.server_v2.api.openai import schema
from mlx_lm.server_v2.chat import tool_call
from mlx_lm.server_v2.engine import protocol

__all__ = ["KEEPALIVE_EVENT", "ChatOutput", "CompletionOutput", "error_body"]

KEEPALIVE_EVENT = ": keepalive\n\n"


class ChatOutput:
    """The chat output of one request, built from engine updates."""

    def __init__(
        self,
        model_id: str,
        *,
        parser: tool_call.ToolCallParser,
        tokenizer: tokenizer_utils.TokenizerWrapper,
        engine_updates: AsyncIterator[protocol.Update],
        include_usage: bool,
        parallel_tool_calls: bool,
    ):
        self._head = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": model_id,
        }

        self._parser = parser
        self._tokenizer = tokenizer
        self._engine_updates = engine_updates
        self._parallel_tool_calls = parallel_tool_calls
        self._num_tool_calls = 0
        self._include_usage = include_usage

    async def stream(self) -> AsyncIterator[str]:
        """The output as server-sent events; an error ends them with an error event."""
        try:
            # OpenAI streams start with a chunk that gives only the role.
            first = schema.ChatDelta(role="assistant", content="")
            yield self._chunk_event(first)

            async for delta, logprobs, update in self._deltas():
                if not delta.is_empty or logprobs:
                    yield self._chunk_event(delta, logprobs=logprobs)

                if (end := update.end) is not None:
                    last = schema.ChatDelta()
                    yield self._chunk_event(last, self._chat_finish_reason(end))
                    if self._include_usage:
                        usage = schema.ChatCompletionChunk(
                            **self._head,
                            object="chat.completion.chunk",
                            choices=[],
                            usage=_usage(end.usage),
                        )
                        yield _sse_event(usage)
        except Exception as e:
            yield _error_event(e)
        yield "data: [DONE]\n\n"

    async def body(self) -> dict[str, Any]:
        """The whole output as one response body."""
        contents, reasonings, openai_calls, logprobs = [], [], [], []
        end = None
        async for delta, delta_logprobs, update in self._deltas():
            end = update.end
            logprobs += delta_logprobs
            contents.append(delta.content or "")
            reasonings.append(delta.reasoning or "")
            # The index is only for streams.
            openai_calls += [
                schema.ToolCall(**call.model_dump(exclude={"index"}))
                for call in delta.tool_calls or []
            ]

        # The updates end with FinishedEvent: a FailedEvent raises in from_events.
        assert end is not None
        reasoning = "".join(reasonings)
        message = schema.ChatMessage(
            role="assistant",
            content="".join(contents) or None,
            **_non_empty(
                reasoning=reasoning,
                reasoning_content=reasoning,
                tool_calls=openai_calls,
            ),
        )
        choice = schema.ChatChoice(
            index=0,
            message=message,
            finish_reason=self._chat_finish_reason(end),
            logprobs=_chat_logprobs(logprobs),
        )
        completion = schema.ChatCompletion(
            **self._head,
            object="chat.completion",
            choices=[choice],
            usage=_usage(end.usage),
        )
        return completion.to_dict()

    async def _deltas(
        self,
    ) -> AsyncIterator[
        tuple[
            schema.ChatDelta,
            list[schema.TokenLogprob],
            protocol.Update,
        ]
    ]:
        """Message deltas and the log probabilities of their tokens, if asked.

        Each comes with its update; the last has the end. An empty delta means nothing
        to send yet. Counts the tool calls that ``_chat_finish_reason`` reads.
        """
        async for update in self._engine_updates:
            content, reasoning, calls, logprobs = "", "", [], []
            for event in update.events:
                # Parse each part: a tool call that stays text keeps its place.
                text, event_calls = self._parser.parse(event.delta)
                content += text
                reasoning += event.delta.reasoning
                calls += event_calls
                logprobs += _token_logprobs(self._tokenizer, event.logprobs)

            if not self._parallel_tool_calls:
                # Only the first call of the response.
                calls = calls[:1] if self._num_tool_calls == 0 else []

            openai_calls = [
                _tool_call(c, index=self._num_tool_calls + i)
                for i, c in enumerate(calls)
            ]
            self._num_tool_calls += len(calls)
            fields = _non_empty(
                content=content,
                reasoning=reasoning,
                reasoning_content=reasoning,
                tool_calls=openai_calls,
            )
            yield schema.ChatDelta(**fields), logprobs, update

    def _chunk_event(
        self,
        delta: schema.ChatDelta,
        finish_reason: schema.ChatFinishReason | None = None,
        *,
        logprobs: Sequence[schema.TokenLogprob] = (),
    ) -> str:
        choice = schema.ChatChunkChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
            **_non_empty(logprobs=_chat_logprobs(logprobs)),
        )

        # The usage is null in the chunks before the last one.
        usage = {"usage": None} if self._include_usage else {}
        chunk = schema.ChatCompletionChunk(
            **self._head, object="chat.completion.chunk", choices=[choice], **usage
        )
        return _sse_event(chunk)

    def _chat_finish_reason(
        self, end: protocol.FinishedEvent
    ) -> schema.ChatFinishReason:
        reason = _finish_reason(end)
        return "tool_calls" if self._num_tool_calls and reason == "stop" else reason


class CompletionOutput:
    """The text completion output of one request, built from engine updates."""

    def __init__(
        self,
        model_id: str,
        *,
        tokenizer: tokenizer_utils.TokenizerWrapper,
        engine_updates: AsyncIterator[protocol.Update],
        include_usage: bool,
    ):
        self._head = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
        }

        self._tokenizer = tokenizer
        self._engine_updates = engine_updates
        self._include_usage = include_usage

    async def stream(self) -> AsyncIterator[str]:
        """The output as server-sent events; an error ends them with an error event."""
        try:
            async for update in self._engine_updates:
                text = "".join(event.delta.content for event in update.events)
                logprobs = self._logprobs(update.events)
                # The schema has no null usage here, unlike chat chunks.
                yield _sse_event(self._completion(text, update.end, logprobs=logprobs))

                if (end := update.end) is not None and self._include_usage:
                    usage = schema.Completion(
                        **self._head, choices=[], usage=_usage(end.usage)
                    )
                    yield _sse_event(usage)
        except Exception as e:
            yield _error_event(e)
        yield "data: [DONE]\n\n"

    async def body(self) -> dict[str, Any]:
        """The whole output as one response body."""
        events, end = [], None
        async for update in self._engine_updates:
            events += update.events
            end = update.end

        assert end is not None
        text = "".join(event.delta.content for event in events)
        logprobs = self._logprobs(events)
        completion = self._completion(
            text, end, logprobs=logprobs, usage=_usage(end.usage)
        )
        return completion.to_dict()

    def _completion(
        self,
        text: str,
        end: protocol.FinishedEvent | None,
        *,
        logprobs: schema.CompletionLogprobs | None = None,
        usage: schema.Usage | None = None,
    ) -> schema.Completion:
        reason = _finish_reason(end) if end is not None else None
        choice = schema.CompletionChoice(
            index=0, text=text, finish_reason=reason, logprobs=logprobs
        )
        return schema.Completion(
            **self._head, choices=[choice], **_non_empty(usage=usage)
        )

    def _logprobs(
        self, events: list[protocol.TokensEvent]
    ) -> schema.CompletionLogprobs | None:
        """The log probabilities of the events' tokens; None without any."""
        tokens, token_logprobs, top_logprobs = [], [], []
        for event in events:
            for entry in _token_logprobs(self._tokenizer, event.logprobs):
                tokens.append(entry.token)
                token_logprobs.append(entry.logprob)
                top = {t.token: t.logprob for t in entry.top_logprobs}
                # Also the sampled token, as the OpenAI API has it.
                top[entry.token] = entry.logprob
                top_logprobs.append(top)

        if not tokens:
            return None
        return schema.CompletionLogprobs(
            tokens=tokens, token_logprobs=token_logprobs, top_logprobs=top_logprobs
        )


def error_body(error: http.APIError) -> dict[str, Any]:
    """An error in the OpenAI format."""
    error_type = "server_error" if error.status >= 500 else "invalid_request_error"
    return {
        "error": {
            "message": str(error),
            "type": error_type,
            "param": error.param,
            "code": error.code,
        }
    }


def _token_logprobs(
    tokenizer: tokenizer_utils.TokenizerWrapper,
    logprobs: Sequence[protocol.TokenLogprob],
) -> list[schema.TokenLogprob]:
    """Log probabilities in the OpenAI chat format."""
    entries = []
    for token_logprob in logprobs:
        entry = _logprob_fields(tokenizer, token_logprob.token, token_logprob.logprob)
        top = [
            schema.TopLogprob(**_logprob_fields(tokenizer, t, logprob))
            for t, logprob in token_logprob.top
        ]
        entries.append(schema.TokenLogprob(**entry, top_logprobs=top))

    return entries


def _chat_logprobs(
    logprobs: Sequence[schema.TokenLogprob],
) -> schema.ChatLogprobs | None:
    if not logprobs:
        return None
    return schema.ChatLogprobs(content=logprobs, refusal=None)


def _logprob_fields(
    tokenizer: tokenizer_utils.TokenizerWrapper, token: int, logprob: float
) -> dict[str, Any]:
    text = tokenizer.decode([token])

    # JSON has no -inf or NaN: send -9999.0, as vLLM does.
    finite_logprob = -9999.0
    if math.isfinite(logprob):
        finite_logprob = max(logprob, -9999.0)

    # A token that ends inside a character has no text of its own.
    token_bytes = None
    if "\ufffd" not in text:
        token_bytes = list(text.encode("utf-8", errors="replace"))

    return {"token": text, "logprob": finite_logprob, "bytes": token_bytes}


def _tool_call(call: tool_call.ToolCall, *, index: int) -> schema.ToolCall:
    """A parsed tool call in the OpenAI form: the arguments are a JSON string."""
    arguments = json.dumps(call.arguments, ensure_ascii=False)
    function = schema.ToolCallFunction(name=call.name, arguments=arguments)
    return schema.ToolCall(index=index, id=call.id, type="function", function=function)


def _finish_reason(end: protocol.FinishedEvent) -> Literal["stop", "length"]:
    # OpenAI has no cancelled reason.
    return "length" if end.reason == protocol.FinishReason.LENGTH else "stop"


def _usage(usage: protocol.Usage) -> schema.Usage:
    return schema.Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.prompt_tokens + usage.completion_tokens,
        prompt_tokens_details=schema.PromptTokensDetails(
            cached_tokens=usage.cached_tokens
        ),
    )


def _non_empty(**fields: Any) -> dict[str, Any]:
    """The fields that have a value: a response sends only these."""
    return {key: value for key, value in fields.items() if value}


def _sse_event(body: schema.BaseResponse) -> str:
    return f"data: {body.to_json()}\n\n"


def _error_event(e: Exception) -> str:
    """The event of an error in the middle of a stream."""
    error = error_body(http.APIError.from_exception(e))
    return f"data: {json.dumps(error)}\n\n"
