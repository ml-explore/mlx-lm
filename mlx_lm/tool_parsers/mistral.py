# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

_tool_call_regex = re.compile(r"\s*(\w+)\[ARGS\]\s*(\{.*\})", re.DOTALL)

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    match = _tool_call_regex.search(text)
    if match is None:
        raise ValueError(f"Could not parse tool call from: {text}")
    func_name = match.group(1)
    func_args = json.loads(match.group(2))
    return dict(name=func_name, arguments=func_args)
