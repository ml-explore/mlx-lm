# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

_tool_call_regex = re.compile(r"\s*(\w+)\[ARGS\]\s*(\{.*\})", re.DOTALL)

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def _parse_json_array(text: str):
    """Parse the classic ``[TOOL_CALLS][{"name": ..., "arguments": {...}}]``
    payload emitted by Mistral Nemo, Mistral Small 2409, Ministral and
    Mixtral v0.3.  Returns ``None`` if ``text`` is not such an array."""
    text = text.strip()
    if not text.startswith("["):
        return None
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list) or not calls:
        return None
    if not all(isinstance(c, dict) and isinstance(c.get("name"), str) for c in calls):
        return None
    parsed = []
    for call in calls:
        tool_call = dict(name=call["name"], arguments=call.get("arguments") or {})
        # Preserve the model's own call id so the follow-up ``tool`` message
        # can be matched back to this call.
        if isinstance(call.get("id"), str):
            tool_call["id"] = call["id"]
        parsed.append(tool_call)
    return parsed


def parse_tool_call(text: str, tools: Any | None = None):
    match = _tool_call_regex.search(text)
    if match is not None:
        func_name = match.group(1)
        func_args = json.loads(match.group(2))
        return dict(name=func_name, arguments=func_args)
    parsed = _parse_json_array(text)
    if parsed is not None:
        return parsed
    raise ValueError(f"Could not parse tool call from: {text}")
