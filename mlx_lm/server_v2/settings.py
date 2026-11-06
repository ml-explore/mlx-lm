# Copyright © 2026 Apple Inc.

from typing import Annotated, Any

import pydantic

__all__ = ["EngineConfig", "GenerationParams", "ServerConfig"]


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    config=pydantic.ConfigDict(strict=True, extra="forbid", allow_inf_nan=False),
)
class GenerationParams:
    """How to sample one request; the defaults decode greedily."""

    # None generates up to the engine's default, within the context length.
    max_tokens: Annotated[int, pydantic.Field(ge=1)] | None = None

    temperature: float = pydantic.Field(0.0, ge=0.0)
    top_p: float = pydantic.Field(1.0, gt=0.0, le=1.0)
    top_k: int = pydantic.Field(0, ge=0)
    min_p: float = pydantic.Field(0.0, ge=0.0, le=1.0)
    xtc_probability: float = pydantic.Field(0.0, ge=0.0, le=1.0)
    xtc_threshold: float = pydantic.Field(0.1, ge=0.0, le=0.5)

    repetition_penalty: Annotated[float, pydantic.Field(ge=0.0)] | None = None
    # The penalties slice the past tokens, and MLX slices take 32-bit indices.
    repetition_context_size: int = pydantic.Field(20, ge=1, le=2**31 - 1)
    presence_penalty: float | None = None
    presence_context_size: int = pydantic.Field(20, ge=1, le=2**31 - 1)
    frequency_penalty: float | None = None
    frequency_context_size: int = pydantic.Field(20, ge=1, le=2**31 - 1)
    logit_bias: dict[Annotated[int, pydantic.Field(ge=0)], float] | None = None

    # At most 16 strings of 256 characters: each is checked at every token.
    # A list is also taken; the strings stay strict.
    stop: Annotated[
        tuple[Annotated[str, pydantic.Field(max_length=256)], ...],
        pydantic.Field(max_length=16, strict=False),
    ] = ()

    # The number of most likely tokens to send with each token's log
    # probability; None sends no log probabilities.
    logprobs: Annotated[int, pydantic.Field(ge=0, le=20)] | None = None


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    config=pydantic.ConfigDict(strict=True, extra="forbid"),
)
class EngineConfig:
    """The engine's limits: the context, the rows, the prefill and the cache store."""

    context_length: Annotated[int, pydantic.Field(ge=2)] | None = None
    max_tokens: int = pydantic.Field(8192, ge=1)

    decode_concurrency: int = pydantic.Field(32, ge=1)
    # Prompt tokens per step.
    prefill_step_size: int = pydantic.Field(512, ge=1)

    prompt_cache_size: int = pydantic.Field(10, ge=0)
    prompt_cache_bytes: int = pydantic.Field(1 << 63, ge=0)

    # Seconds without an event before a request gets an idle response.
    keepalive: float = pydantic.Field(15.0, gt=0.0)


class ServerConfig(pydantic.BaseModel):
    """The served model, and the defaults of its requests."""

    model_config = pydantic.ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str
    default_params: GenerationParams = GenerationParams()
    template_kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    stall_timeout: float = pydantic.Field(60.0, gt=0.0)
