# Copyright © 2025 Apple Inc.

import json
from typing import Any, Optional

import regex as re

# Matches call:name{...} with balanced braces via the regex module's
# recursive (?R)-style support.  (\{(?:[^{}]|(?2))*\}) recurses on the
# second capture group so nested objects like {a:{b:1}} are captured whole.
_tool_call_regex = re.compile(r"call:(\w+)(\{(?:[^{}]|(?2))*\})", re.DOTALL)


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


def _parse_single(match: re.Match) -> dict:
    """Parse a single call:name{args} regex match into a tool call dict."""
    func_name = match.group(1)
    args_str = match.group(2)
    json_str = _gemma4_args_to_json(args_str)
    arguments = json.loads(json_str)
    return dict(name=func_name, arguments=arguments)


def parse_tool_call(text: str, _: Optional[Any] = None):
    matches = list(_tool_call_regex.finditer(text))
    if not matches:
        # No `call:name{...}` pattern in the output — this happens on every
        # plain-text reply when the request had `tools=[...]` attached.
        # Return an empty list instead of raising so the HTTP server path
        # can emit a normal chat-completion response rather than a dropped
        # stream (see #1125).
        return []
    if len(matches) == 1:
        return _parse_single(matches[0])
    return [_parse_single(m) for m in matches]


tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"
