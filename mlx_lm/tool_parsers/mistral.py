# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

# Matches a "name[ARGS]" header, ending where the JSON arguments start.
_tool_call_header_regex = re.compile(r"([\w-]+)\s*\[ARGS\]\s*")

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def _parse_json_calls(text: str) -> list[dict[str, Any]]:
    """Parse `[{"name": ..., "arguments": ...}]`, raising if it is JSON but
    not a tool call, so it cannot fall through to the header parser."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    calls = []
    for call in payload:
        name = call.get("name") if isinstance(call, dict) else None
        arguments = call.get("arguments", {}) if isinstance(call, dict) else None
        if isinstance(arguments, str):
            # Some templates render the arguments as a JSON string.
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError(f"Could not parse tool call from: {text}")
        parsed = dict(name=name, arguments=arguments)
        if isinstance(call.get("id"), str):
            # The server prefers the model's id over a fresh uuid.
            parsed["id"] = call["id"]
        calls.append(parsed)
    return calls


def _parse_header_calls(text: str) -> list[dict[str, Any]]:
    """Parse `name[ARGS]{...}`, which can repeat. The template always renders
    valid JSON, so a decode failure means generation was cut short."""
    # raw_decode reads exactly one JSON value and reports where it ended.
    decoder = json.JSONDecoder()
    calls = []
    pos = 0
    while (match := _tool_call_header_regex.search(text, pos)) is not None:
        pos = match.end()
        try:
            arguments, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse tool call from: {text}") from e
        calls.append(dict(name=match.group(1), arguments=arguments))
    return calls


def parse_tool_call(text: str, tools: Any | None = None):
    # Mistral has no tool-call end token, so the text can hold several calls.
    calls = _parse_json_calls(text) or _parse_header_calls(text)
    if not calls:
        raise ValueError(f"Could not parse tool call from: {text}")
    return calls[0] if len(calls) == 1 else calls
