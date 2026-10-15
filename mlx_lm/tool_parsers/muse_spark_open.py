# Copyright © 2026 Apple Inc.

"""Tool call parser for Muse Spark Open (XML_ATEM)."""

import json
from typing import Any, Optional

import regex as re

tool_call_start = "<atem:function_calls>"
tool_call_end = "</atem:function_calls>"

_invoke_regex = re.compile(
    r"<atem:invoke\s+name=[\"']?(?P<name>[^\"'>]+)[\"']?\s*>"
    r"(?P<body>(?:<atem:parameter\b.*?</atem:parameter>|.)*?)"
    r"(?:</atem:invoke>|$)",
    re.DOTALL,
)
_param_regex = re.compile(
    r"<atem:parameter\s+name=[\"']?(?P<name>[A-Za-z0-9_.-]+)[\"']?[^>]*>"
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


def _param_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def parse_tool_call(
    model_output: str, tools: Optional[Any] = None
) -> dict[str, Any] | list[dict[str, Any]]:
    calls = []
    for invoke in _invoke_regex.finditer(model_output):
        arguments = {
            param.group("name"): _param_value(param.group("value"))
            for param in _param_regex.finditer(invoke.group("body"))
        }
        calls.append({"name": invoke.group("name"), "arguments": arguments})
    if not calls:
        raise ValueError("No function provided.")
    if len(calls) == 1:
        return calls[0]
    return calls
