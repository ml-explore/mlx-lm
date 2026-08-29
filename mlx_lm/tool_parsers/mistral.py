# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

_tool_call_regex = re.compile(r"\s*(\w+)\[ARGS\]\s*(\{.*\})", re.DOTALL)

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def _parse_json_list(text: str):
    try:
        tool_calls = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse tool call from: {text}") from e

    if not isinstance(tool_calls, list):
        raise ValueError(f"Could not parse tool call from: {text}")

    parsed = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError(f"Could not parse tool call from: {text}")

        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        if not isinstance(name, str):
            raise ValueError(f"Could not parse tool call from: {text}")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse tool call from: {text}") from e

        if not isinstance(arguments, dict):
            raise ValueError(f"Could not parse tool call from: {text}")

        result = {"name": name, "arguments": arguments}
        if "id" in tool_call:
            if not isinstance(tool_call["id"], str):
                raise ValueError(f"Could not parse tool call from: {text}")
            result["id"] = tool_call["id"]
        parsed.append(result)

    return parsed


def parse_tool_call(text: str, tools: Any | None = None):
    match = _tool_call_regex.search(text)
    if match is None:
        return _parse_json_list(text)
    func_name = match.group(1)
    func_args = json.loads(match.group(2))
    return dict(name=func_name, arguments=func_args)
