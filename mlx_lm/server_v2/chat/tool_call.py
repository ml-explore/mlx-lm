# Copyright © 2026 Apple Inc.

import json
import logging
import secrets
import string
from typing import Annotated, Any, Self

import pydantic

from mlx_lm import tokenizer_utils
from mlx_lm.generate import TextStateMachine

__all__ = ["ToolCall", "ToolCallParser"]

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, config=pydantic.ConfigDict(strict=True)
)
class ToolCall:
    """A parsed tool call: its id, its function name and its arguments."""

    id: str
    name: Annotated[str, pydantic.Field(min_length=1)]
    # An object: the template of the next request needs one.
    arguments: dict[str, Any]

    @classmethod
    def from_parsed(cls, tool_call: dict[str, Any]) -> Self:
        """The tool call that a parser returns; raises ``ValueError`` if it is bad."""
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        # The APIs send the arguments as JSON, which has no NaN or Infinity.
        _ = json.dumps(arguments, allow_nan=False)

        # Some parsers take the id from the model text.
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = _tool_call_id()

        return cls(id=call_id, name=tool_call["name"], arguments=arguments)


class ToolCallParser:
    """Parse the tool calls of the engine's text deltas."""

    def __init__(
        self,
        tokenizer: tokenizer_utils.TokenizerWrapper,
        *,
        tools: list[dict[str, Any]] | None = None,
    ):
        self._parser = tokenizer.tool_parser
        self._start = tokenizer.tool_call_start or ""
        self._end = tokenizer.tool_call_end or ""
        self._tool_definitions = tools

    def parse(self, delta: TextStateMachine.Delta) -> tuple[str, list[ToolCall]]:
        """The content and the parsed tool calls of ``delta``."""
        content = delta.content
        calls = []
        for text in delta.tool_calls:
            if self._tool_definitions is None:
                content += self._start + text + self._end
                continue

            try:
                parsed = self._parser(text, self._tool_definitions)
                parsed = parsed if isinstance(parsed, list) else [parsed]
                if not parsed:
                    raise ValueError("The block has no tool calls.")
                # All calls of a block parse, or none of them.
                calls += [ToolCall.from_parsed(tc) for tc in parsed]
            except Exception as e:
                # A call that does not parse stays in the content.
                _logger.warning(
                    "A tool call did not parse (%s: %s).", type(e).__name__, e
                )
                content += self._start + text + self._end

        return content, calls


def _tool_call_id() -> str:
    # Nine letters and digits fit every chat template, Mistral's included.
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(9))
