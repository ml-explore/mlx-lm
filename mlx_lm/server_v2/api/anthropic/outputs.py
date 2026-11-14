# Copyright © 2026 Apple Inc.

import itertools
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, assert_never

from mlx_lm.server_v2.api import http
from mlx_lm.server_v2.api.anthropic import schema
from mlx_lm.server_v2.chat import tool_call
from mlx_lm.server_v2.engine import protocol

__all__ = ["PING_EVENT", "MessageOutput", "context_length_error", "error_body"]

PING_EVENT = 'event: ping\ndata: {"type":"ping"}\n\n'


class MessageOutput:
    """The message output of one request, built from engine updates."""

    def __init__(
        self,
        model_id: str,
        *,
        parser: tool_call.ToolCallParser,
        engine_updates: AsyncIterator[protocol.Update],
        input_tokens: int,
        limited_by_context: bool,
    ):
        self._head = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model_id,
        }

        self._parser = parser
        self._engine_updates = engine_updates
        # The prompt tokens; the end tells how many were cached.
        self._input_tokens = input_tokens
        # Whether max_tokens was cut to the room left in the context.
        self._limited_by_context = limited_by_context
        self._num_tool_calls = 0

    async def stream(self) -> AsyncIterator[str]:
        """The output as server-sent events; an error ends them with an error event."""
        try:
            # The cached tokens are known only at the end: message_delta has them.
            usage = schema.Usage(
                input_tokens=self._input_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                output_tokens=0,
            )
            message = schema.Message(
                **self._head,
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=usage,
            )
            start = schema.MessageStartEvent(type="message_start", message=message)
            yield _sse_event(start)

            blocks = _BlockStream()
            async for parts, update in self._parts():
                for part in parts:
                    for event in blocks.add(part):
                        yield _sse_event(event)

                if (end := update.end) is not None:
                    for event in blocks.close():
                        yield _sse_event(event)
                    yield _sse_event(self._message_delta(end))
                    yield _sse_event(schema.MessageStopEvent(type="message_stop"))
        except Exception as e:
            yield _error_event(e)

    async def body(self) -> dict[str, Any]:
        """The whole output as one response body."""
        parts, end = [], None
        async for update_parts, update in self._parts():
            parts += update_parts
            end = update.end

        # The updates end with FinishedEvent: a FailedEvent raises in from_events.
        assert end is not None
        stop_reason, stop_sequence = self._stop(end)
        message = schema.Message(
            **self._head,
            content=_join_parts(parts),
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=_usage(end.usage),
        )
        return message.model_dump()

    async def _parts(
        self,
    ) -> AsyncIterator[tuple[list[schema.ContentBlock], protocol.Update]]:
        """The content parts of each update, with the update; the last has the end.

        A part is a content block, or more text of the block before it. Counts the
        tool calls that ``_stop`` reads.
        """
        async for update in self._engine_updates:
            parts = []
            for event in update.events:
                # Parse each event: a tool call that stays text keeps its place.
                text, calls = self._parser.parse(event.delta)
                self._num_tool_calls += len(calls)

                if event.delta.reasoning:
                    parts.append(
                        schema.ThinkingBlock(
                            type="thinking",
                            thinking=event.delta.reasoning,
                            signature="",
                        )
                    )
                if text:
                    parts.append(schema.TextBlock(type="text", text=text))
                parts += [_tool_use_block(call) for call in calls]
            yield parts, update

    def _message_delta(self, end: protocol.FinishedEvent) -> schema.MessageDeltaEvent:
        """The event that ends the message, with its stop reason and usage."""
        stop_reason, stop_sequence = self._stop(end)
        delta = schema.MessageDelta(
            stop_reason=stop_reason, stop_sequence=stop_sequence
        )
        return schema.MessageDeltaEvent(
            type="message_delta", delta=delta, usage=_usage(end.usage)
        )

    def _stop(
        self, end: protocol.FinishedEvent
    ) -> tuple[schema.StopReason, str | None]:
        """The stop reason, and the stop sequence that ended the reply if one did."""
        if end.reason == protocol.FinishReason.LENGTH:
            if self._limited_by_context:
                return "model_context_window_exceeded", None
            return "max_tokens", None
        # Also after a stop sequence: the client must run the calls.
        if self._num_tool_calls:
            return "tool_use", None
        if end.reason == protocol.FinishReason.STOP_STRING:
            return "stop_sequence", end.stop_string
        # Anthropic has no cancelled reason.
        return "end_turn", None


class _BlockStream:
    """The content block events of a stream, made from the message parts."""

    def __init__(self):
        self._index = -1
        self._open_type: str | None = None

    def add(self, part: schema.ContentBlock) -> list[schema.StreamEvent]:
        """The events of a part: more text of the open block, or a new block."""
        if part.type == self._open_type:
            return [self._delta(part)]

        events = self.close()
        self._index += 1
        self._open_type = part.type
        start = schema.ContentBlockStartEvent(
            type="content_block_start",
            index=self._index,
            content_block=_empty_block(part),
        )
        events += [start, self._delta(part)]
        if part.type == "tool_use":
            # The input comes in one part: the block ends with it.
            events += self.close()
        return events

    def close(self) -> list[schema.StreamEvent]:
        """The event that ends the open block; none if no block is open."""
        if self._open_type is None:
            return []
        self._open_type = None
        return [
            schema.ContentBlockStopEvent(type="content_block_stop", index=self._index)
        ]

    def _delta(self, part: schema.ContentBlock) -> schema.ContentBlockDeltaEvent:
        return schema.ContentBlockDeltaEvent(
            type="content_block_delta", index=self._index, delta=_block_delta(part)
        )


def error_body(error: http.APIError) -> dict[str, Any]:
    """An error in the Anthropic format."""
    error_type = "api_error" if error.status >= 500 else "invalid_request_error"
    return {"type": "error", "error": {"type": error_type, "message": str(error)}}


def context_length_error(num_tokens: int, context_length: int) -> http.APIError:
    """The 400 for a prompt that does not fit, in the words of the Anthropic API.

    Claude Code reads these words: it compacts a long prompt.
    """
    # The reply needs at least one token.
    message = f"prompt is too long: {num_tokens} tokens > {context_length - 1} maximum"
    return http.APIError(400, message)


def _tool_use_block(call: tool_call.ToolCall) -> schema.ToolUseBlock:
    """A parsed tool call as a tool_use block."""
    return schema.ToolUseBlock(
        type="tool_use", id=call.id, name=call.name, input=call.arguments
    )


def _join_parts(parts: list[schema.ContentBlock]) -> list[schema.ContentBlock]:
    """The blocks of the parts: a run of text or thinking parts is one block."""
    blocks = []
    for kind, run in itertools.groupby(parts, key=type):
        match kind:
            case schema.TextBlock:
                text = "".join(part.text for part in run)
                blocks.append(schema.TextBlock(type="text", text=text))
            case schema.ThinkingBlock:
                thinking = "".join(part.thinking for part in run)
                blocks.append(
                    schema.ThinkingBlock(
                        type="thinking", thinking=thinking, signature=""
                    )
                )
            case schema.ToolUseBlock:
                blocks += run
            case _:
                assert_never(kind)
    return blocks


def _empty_block(part: schema.ContentBlock) -> schema.ContentBlock:
    """The block of a part without its text or input."""
    match part:
        case schema.TextBlock():
            return schema.TextBlock(type="text", text="")
        case schema.ThinkingBlock():
            return schema.ThinkingBlock(type="thinking", thinking="", signature="")
        case schema.ToolUseBlock():
            return schema.ToolUseBlock(
                type="tool_use", id=part.id, name=part.name, input={}
            )


def _block_delta(
    part: schema.ContentBlock,
) -> schema.TextDelta | schema.ThinkingDelta | schema.InputJSONDelta:
    """The delta of a part: its text, or the JSON of its input."""
    match part:
        case schema.TextBlock():
            return schema.TextDelta(type="text_delta", text=part.text)
        case schema.ThinkingBlock():
            return schema.ThinkingDelta(type="thinking_delta", thinking=part.thinking)
        case schema.ToolUseBlock():
            partial_json = json.dumps(part.input, ensure_ascii=False)
            return schema.InputJSONDelta(
                type="input_json_delta", partial_json=partial_json
            )


def _usage(usage: protocol.Usage) -> schema.Usage:
    """The usage of the engine; the input tokens exclude the cached ones."""
    return schema.Usage(
        input_tokens=usage.prompt_tokens - usage.cached_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=usage.cached_tokens,
        output_tokens=usage.completion_tokens,
    )


def _sse_event(event: schema.StreamEvent) -> str:
    """An event as a server-sent event."""
    # The Anthropic SDK skips an event without an "event:" line.
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _error_event(e: Exception) -> str:
    """The event of an error in the middle of a stream."""
    error = error_body(http.APIError.from_exception(e))
    return f"event: error\ndata: {json.dumps(error)}\n\n"
