# Copyright © 2025 Apple Inc.

import json
from typing import Any, Optional

import regex as re

# Matches call:name{...} with balanced braces via the regex module's
# recursive (?R)-style support.  (\{(?:[^{}]|(?2))*\}) recurses on the
# second capture group so nested objects like {a:{b:1}} are captured whole.
_tool_call_regex = re.compile(r"call:(\w+)(\{(?:[^{}]|(?2))*\})", re.DOTALL)

_STRING_DELIM = '<|"|>'


def _shield_string_braces(text: str) -> str:
    """Replace ``{`` and ``}`` inside <|"|>…<|"|> spans with placeholders
    so the balanced-brace regex is not confused by string content."""
    parts: list[str] = []
    i = 0
    n = len(text)
    dlen = len(_STRING_DELIM)
    while i < n:
        if text[i : i + dlen] == _STRING_DELIM:
            end = text.find(_STRING_DELIM, i + dlen)
            if end == -1:
                parts.append(text[i:])
                break
            inner = text[i + dlen : end].replace("{", "\x01").replace("}", "\x02")
            parts.append(_STRING_DELIM + inner + _STRING_DELIM)
            i = end + dlen
        else:
            parts.append(text[i])
            i += 1
    return "".join(parts)


def _restore_braces(text: str) -> str:
    """Restore shielded brace placeholders."""
    return text.replace("\x01", "{").replace("\x02", "}")


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
    args_str = _restore_braces(match.group(2))
    json_str = _gemma4_args_to_json(args_str)
    arguments = json.loads(json_str)
    return dict(name=func_name, arguments=arguments)


def parse_tool_call(text: str, _: Optional[Any] = None):
    shielded = _shield_string_braces(text)
    matches = list(_tool_call_regex.finditer(shielded))
    if not matches:
        raise ValueError("No function provided.")
    if len(matches) == 1:
        return _parse_single(matches[0])
    return [_parse_single(m) for m in matches]


tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"
