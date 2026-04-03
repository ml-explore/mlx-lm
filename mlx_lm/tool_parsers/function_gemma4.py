# Copyright © 2025 Apple Inc.

"""Gemma 4 tool call parser.

Gemma 4 uses <|tool_call>...<tool_call|> delimiters with JSON content.
String values use <|"|> as escaped quotes instead of standard \".

Example:
  <|tool_call>{"name": "get_weather", "arguments": {"city": <|"|>London<|"|>}}<tool_call|>

Distinct from Gemma 3's <start_function_call>call:func{key:value} format.
"""

import json
from typing import Any, Optional

import regex as re

_tool_call_regex = re.compile(r"<\|tool_call\>(.*?)<tool_call\|>", re.DOTALL)


def _unescape_quotes(s: str) -> str:
    """Replace Gemma 4's <|"|> escape sequences with standard quotes."""
    return s.replace('<|"|>', '"')


def parse_tool_call(text: str, _: Optional[Any] = None):
    match = _tool_call_regex.findall(text)
    if not match:
        raise ValueError("No tool call found.")

    raw = _unescape_quotes(match[0].strip())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse tool call JSON: {raw}")

    name = parsed.get("name", "")
    arguments = parsed.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass

    return dict(name=name, arguments=arguments)


tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"
