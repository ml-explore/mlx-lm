# Copyright © 2026 Apple Inc.

import asyncio
import functools
from collections.abc import AsyncIterator

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import tokenizer_utils, utils
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.engine import core, errors, protocol

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"

# Like Qwen3: reasoning shows after the last user turn; "think" opens a block.
CHAT_TEMPLATE = (
    "{% set ns = namespace(last=0) %}{% for m in messages %}"
    "{% if m.role == 'user' %}{% set ns.last = loop.index0 %}{% endif %}"
    "{% endfor %}{% for m in messages %}"
    "{% if loop.index0 > ns.last and m.reasoning_content %}"
    "<think>{{ m.reasoning_content }}</think>{% endif %}"
    "{{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt and think %}<think>{% endif %}"
)


@functools.cache
def load_model() -> tuple[nn.Module, tokenizer_utils.TokenizerWrapper]:
    model, tokenizer = utils.load(HF_MODEL_PATH)
    model.set_dtype(mx.float32)
    return model, tokenizer


@functools.cache
def load_tokenizer(**config: str) -> tokenizer_utils.TokenizerWrapper:
    return utils.load_tokenizer(HF_MODEL_PATH, config)


@functools.cache
def load_chat_tokenizer() -> tokenizer_utils.TokenizerWrapper:
    """A tokenizer with reasoning and tool calls."""
    return load_tokenizer(chat_template=CHAT_TEMPLATE, tool_parser_type="json_tools")


class StubEngine:
    """Serves ``events`` to each request as one response.

    A request that has no end waits, with idle updates, until its updates close.
    """

    def __init__(self, *events: protocol.Event):
        self.config = settings.EngineConfig(context_length=4096)
        self.status = core.EngineStatus.OK
        self.error = None
        self.stalled_for = 0.0
        self.events = events
        self.requests = []
        self.cancelled = []
        self.submitted = asyncio.Event()
        self.cancel_called = asyncio.Event()
        self.started = False

    async def start(self) -> None:
        self.started = True
        self.status = core.EngineStatus.OK

    async def close(self) -> None:
        self.status = core.EngineStatus.CLOSED

    def submit(self, request: protocol.Request) -> int:
        # Like Engine.submit: raise at once, not in the updates.
        if self.status != core.EngineStatus.OK:
            raise errors.UnavailableError(self.status)
        self.requests.append(request)
        self.submitted.set()
        return len(self.requests) - 1

    def updates(self, handle: int) -> AsyncIterator[protocol.Update]:
        return self._updates(handle)

    async def _updates(self, handle: int) -> AsyncIterator[protocol.Update]:
        try:
            if self.events:
                yield protocol.Update.from_events(list(self.events))
            if not self.events or isinstance(self.events[-1], protocol.TokensEvent):
                while True:
                    await asyncio.sleep(self.config.keepalive)
                    yield protocol.Update([])
        finally:
            self.cancelled.append(handle)
            self.cancel_called.set()

    def seconds_since_step(self) -> float:
        return self.stalled_for
