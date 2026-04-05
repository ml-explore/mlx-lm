# Copyright © 2025 Apple Inc.

import json
import re
from typing import Any, Optional

_tool_call_regex = re.compile(r"call:(\w+)(\{.*\})", re.DOTALL)


def _gemma4_args_to_json(text: str) -> str:
    """Convert Gemma 4 tool call args to valid JSON.

    Gemma 4 uses unquoted keys and <|"|> as string delimiters
    instead of standard double quotes.
    """
    strings = []

    def _capture(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    # Extract <|"|>-delimited strings and replace with placeholders
    text = re.sub(r'<\|"\|>(.*?)<\|"\|>', _capture, text, flags=re.DOTALL)
    # Quote bare keys
    text = re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)
    # Restore captured strings as properly escaped JSON strings
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s))

    return text


def parse_tool_call(text: str, _: Optional[Any] = None):
    match = _tool_call_regex.search(text)
    if not match:
        raise ValueError("No function provided.")
    func_name = match.group(1)
    args_str = match.group(2)
    json_str = _gemma4_args_to_json(args_str)
    arguments = json.loads(json_str)
    return dict(name=func_name, arguments=arguments)


tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"
