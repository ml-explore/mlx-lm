# Copyright © 2026 Apple Inc.

"""Tool parser for IFM K2-Horizon models."""

import ast
import json
import re
from typing import Any

tool_call_start = "<ifm|tool_calls>"
tool_call_end = "</ifm|tool_calls>"

_tool_call_re = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.DOTALL)
_argument_re = re.compile(
    r"<ifm\|arg_key>\s*(.*?)\s*</ifm\|arg_key>\s*"
    r"(?:<ifm\|arg_type>\s*.*?\s*</ifm\|arg_type>\s*)?"
    r"<ifm\|arg_value>\s*(.*?)\s*</ifm\|arg_value>",
    re.DOTALL,
)


def _deserialize(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        pass

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _parse_one(text: str) -> dict[str, Any]:
    matches = list(_argument_re.finditer(text))
    name = text[: matches[0].start()].strip() if matches else text.strip()
    if not name:
        raise ValueError("K2-Horizon tool call is missing a function name")

    arguments = {
        match.group(1).strip(): _deserialize(match.group(2)) for match in matches
    }
    return {"name": name.splitlines()[0].strip(), "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse K2-Horizon's IFM XML-like tool-call format."""
    calls = _tool_call_re.findall(text) or [text]
    return [_parse_one(call) for call in calls]
