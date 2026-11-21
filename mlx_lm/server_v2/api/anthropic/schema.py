# Copyright © 2026 Apple Inc.

from typing import Annotated, Any, Literal

import pydantic

__all__ = [
    "ContentBlock",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "CountTokensRequest",
    "InputJSONDelta",
    "Message",
    "MessageDelta",
    "MessageDeltaEvent",
    "MessageStartEvent",
    "MessageStopEvent",
    "MessageTokensCount",
    "MessagesRequest",
    "RequestContentBlock",
    "RequestMediaBlock",
    "RequestMessage",
    "RequestOutputConfig",
    "RequestRedactedThinkingBlock",
    "RequestTextBlock",
    "RequestThinking",
    "RequestThinkingBlock",
    "RequestTool",
    "RequestToolChoice",
    "RequestToolResultBlock",
    "RequestToolResultContent",
    "RequestToolUseBlock",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolUseBlock",
    "Usage",
]


# The request fields that the server does not use, e.g. cache_control, are ignored.
# Lists come first in the unions: a bad block then gets its own 400 message.
class RequestTextBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    type: Literal["text"]
    text: str


class RequestMediaBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    type: Literal["image", "document"]


class RequestThinkingBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    type: Literal["thinking"]
    thinking: str


class RequestRedactedThinkingBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    # Dropped: its data is encrypted, so no template can show it.
    type: Literal["redacted_thinking"]


class RequestToolUseBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


RequestToolResultContent = Annotated[
    RequestTextBlock | RequestMediaBlock, pydantic.Field(discriminator="type")
]


class RequestToolResultBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    type: Literal["tool_result"]
    tool_use_id: str
    content: list[RequestToolResultContent] | str = ""


# Other blocks fail the validation.
RequestContentBlock = Annotated[
    RequestTextBlock
    | RequestMediaBlock
    | RequestThinkingBlock
    | RequestRedactedThinkingBlock
    | RequestToolUseBlock
    | RequestToolResultBlock,
    pydantic.Field(discriminator="type"),
]


class RequestMessage(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: list[RequestContentBlock] | str


class RequestTool(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    # Only custom tools: Anthropic runs its server tools itself.
    type: Literal["custom"] | None = None
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class RequestToolChoice(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    type: Literal["auto", "any", "tool", "none"]
    disable_parallel_tool_use: bool | None = None


class RequestThinking(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    # budget_tokens is ignored: the engine has no limit for the reasoning tokens.
    type: Literal["enabled", "disabled", "adaptive"]


class RequestOutputConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    format: dict[str, Any] | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class CountTokensRequest(pydantic.BaseModel):
    """The fields that make the prompt."""

    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    model: str | None = None
    messages: list[RequestMessage] = pydantic.Field(min_length=1)
    system: list[RequestTextBlock] | str | None = None
    tools: list[RequestTool] | None = None
    tool_choice: RequestToolChoice | None = None
    thinking: RequestThinking | None = None
    output_config: RequestOutputConfig | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class MessagesRequest(CountTokensRequest):
    """The fields that make the prompt, and the generation fields."""

    # GenerationParams checks the values.
    max_tokens: int
    stream: bool | None = None
    stop_sequences: list[str] | None = None

    # Unset sampling fields take the server defaults.
    # Not capped at 1 as in the Anthropic API.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


class TextBlock(pydantic.BaseModel):
    type: Literal["text"]
    text: str


class ThinkingBlock(pydantic.BaseModel):
    type: Literal["thinking"]
    thinking: str
    # Empty: only Anthropic can sign its thinking.
    signature: str


class ToolUseBlock(pydantic.BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock

StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "model_context_window_exceeded",
]


class Usage(pydantic.BaseModel):
    # The three input counts add up to the prompt tokens.
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int


class Message(pydantic.BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[ContentBlock]
    # Null only in the message_start event of a stream.
    stop_reason: StopReason | None
    stop_sequence: str | None
    usage: Usage


class MessageTokensCount(pydantic.BaseModel):
    input_tokens: int


class TextDelta(pydantic.BaseModel):
    type: Literal["text_delta"]
    text: str


class ThinkingDelta(pydantic.BaseModel):
    type: Literal["thinking_delta"]
    thinking: str


class InputJSONDelta(pydantic.BaseModel):
    type: Literal["input_json_delta"]
    # The JSON text of the tool input.
    partial_json: str


class MessageDelta(pydantic.BaseModel):
    stop_reason: StopReason
    stop_sequence: str | None


class MessageStartEvent(pydantic.BaseModel):
    type: Literal["message_start"]
    message: Message


class ContentBlockStartEvent(pydantic.BaseModel):
    type: Literal["content_block_start"]
    index: int
    # Without its text or input: the deltas bring them.
    content_block: ContentBlock


class ContentBlockDeltaEvent(pydantic.BaseModel):
    type: Literal["content_block_delta"]
    index: int
    delta: TextDelta | ThinkingDelta | InputJSONDelta


class ContentBlockStopEvent(pydantic.BaseModel):
    type: Literal["content_block_stop"]
    index: int


class MessageDeltaEvent(pydantic.BaseModel):
    type: Literal["message_delta"]
    delta: MessageDelta
    usage: Usage


class MessageStopEvent(pydantic.BaseModel):
    type: Literal["message_stop"]


StreamEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
)
