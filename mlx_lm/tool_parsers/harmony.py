# Copyright © 2026 Apple Inc.

"""Tool calls in gpt-oss harmony messages.

The recipient is ``functions.<name>``. The checkpoint renders it before the
channel (``to=functions.<name><|channel|>commentary json<|message|>``). The
model also emits the channel first
(``<|channel|>commentary to=functions.<name> <|constrain|>json<|message|>``).
The response format starts a call at ``to=functions.``, so this parser sees
the name and the JSON payload.
"""

import json
import re
from typing import Any

tool_call_start = "to=functions."
tool_call_end = "<|call|>"

# Name, then whichever of the channel / constrain / content-type the render
# left in front of the payload.
_call_regex = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:<\|channel\|>(?:commentary|analysis))?"
    r"(?:\s*<\|constrain\|>)?"
    # ``json`` may sit against <|constrain|> or after a space, as the template writes it.
    r"(?:\s*[A-Za-z][A-Za-z0-9_]*)?" r"\s*<\|message\|>\s*(?P<args>.*)$",
    re.DOTALL,
)


def _arguments(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(value, dict):
        raise ValueError("Tool call arguments must be a JSON object.")
    return value


def parse_tool_call(text: str, tools: Any | None = None) -> dict[str, Any]:
    match = _call_regex.match(text.strip())
    if match is None:
        raise ValueError("No tool call found.")
    return {"name": match.group("name"), "arguments": _arguments(match.group("args"))}
