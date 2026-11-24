# Copyright © 2026 Apple Inc.

from typing import Annotated, Any, Literal

import pydantic

__all__ = [
    "BaseRequest",
    "BaseResponse",
    "ChatChoice",
    "ChatChunkChoice",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatDelta",
    "ChatFinishReason",
    "ChatLogprobs",
    "ChatMessage",
    "ChatRequest",
    "Completion",
    "CompletionChoice",
    "CompletionLogprobs",
    "CompletionRequest",
    "ModelCard",
    "ModelList",
    "PromptTokensDetails",
    "RequestContentPart",
    "RequestMessage",
    "RequestStreamOptions",
    "RequestToolCall",
    "RequestToolCallFunction",
    "TokenLogprob",
    "ToolCall",
    "ToolCallFunction",
    "TopLogprob",
    "Usage",
]


class RequestStreamOptions(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    include_usage: bool | None = None


# A message keeps its other fields: templates read them, e.g. tool_call_id.
class RequestContentPart(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="allow")

    type: str
    text: str = ""


class RequestToolCallFunction(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="allow")

    name: str
    arguments: str | dict[str, Any] = ""


class RequestToolCall(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="allow")

    function: RequestToolCallFunction


class RequestMessage(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="allow")

    role: str
    content: str | list[RequestContentPart] | None = None
    tool_calls: list[RequestToolCall] | None = None


class BaseRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    model: str | None = None
    stream: bool | None = None
    stream_options: RequestStreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = None
    # The OpenAI API limits the bias of a token.
    logit_bias: dict[str, Annotated[float, pydantic.Field(ge=-100, le=100)]] | None = (
        None
    )

    # The sampling fields: unset ones take the server defaults, and
    # GenerationParams checks the values.
    max_tokens: int | None = None
    # Not capped at 2 as in the OpenAI API.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None

    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    # The OpenAI API limits these two.
    presence_penalty: float | None = pydantic.Field(None, ge=-2.0, le=2.0)
    presence_context_size: int | None = None
    frequency_penalty: float | None = pydantic.Field(None, ge=-2.0, le=2.0)
    frequency_context_size: int | None = None


class ChatRequest(BaseRequest):
    messages: list[RequestMessage] = pydantic.Field(min_length=1)
    max_completion_tokens: int | None = pydantic.Field(None, ge=1)

    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = pydantic.Field(None, ge=0, le=20)

    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionRequest(BaseRequest):
    prompt: str | list[str] | list[int] | list[list[int]]
    echo: bool | None = None
    suffix: str | None = None
    best_of: int | None = None
    logprobs: int | None = pydantic.Field(None, le=5)


class BaseResponse(pydantic.BaseModel):
    """The base of the top-level responses.

    A field without a default is always sent. A field with one is sent only when
    it is set.
    """

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_unset=True)


class PromptTokensDetails(pydantic.BaseModel):
    cached_tokens: int


class Usage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails


class ToolCallFunction(pydantic.BaseModel):
    name: str
    arguments: str


class ToolCall(pydantic.BaseModel):
    # Only in the deltas of a stream.
    index: int | None = None
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class TopLogprob(pydantic.BaseModel):
    token: str
    logprob: float
    # None for a token that ends inside a character.
    bytes: list[int] | None


class TokenLogprob(TopLogprob):
    top_logprobs: list[TopLogprob]


class ChatLogprobs(pydantic.BaseModel):
    content: list[TokenLogprob]
    refusal: None


ChatFinishReason = Literal["stop", "length", "tool_calls"]


class ChatMessage(pydantic.BaseModel):
    role: Literal["assistant"]
    # Null without text, the way the schema has it.
    content: str | None
    reasoning: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatChoice(pydantic.BaseModel):
    index: int
    message: ChatMessage
    finish_reason: ChatFinishReason
    logprobs: ChatLogprobs | None


class ChatCompletion(BaseResponse):
    id: str
    created: int
    model: str
    object: Literal["chat.completion"]
    choices: list[ChatChoice]
    usage: Usage


class ChatDelta(pydantic.BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @property
    def is_empty(self) -> bool:
        """Whether the delta has nothing to send."""
        return not self.model_fields_set


class ChatChunkChoice(pydantic.BaseModel):
    index: int
    delta: ChatDelta
    finish_reason: ChatFinishReason | None
    logprobs: ChatLogprobs | None = None


class ChatCompletionChunk(BaseResponse):
    id: str
    created: int
    model: str
    object: Literal["chat.completion.chunk"]
    choices: list[ChatChunkChoice]
    # Set (to null, or the usage in the last chunk) only with include_usage.
    usage: Usage | None = None


class CompletionLogprobs(pydantic.BaseModel):
    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]


class CompletionChoice(pydantic.BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"] | None
    logprobs: CompletionLogprobs | None


class Completion(BaseResponse):
    """A completion, or a chunk of its stream: the stream has no usage until the end."""

    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class ModelCard(BaseResponse):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    # Not in the OpenAI API, but vLLM and SGLang send it too.
    max_model_len: int | None = None


class ModelList(BaseResponse):
    object: Literal["list"]
    data: list[ModelCard]
