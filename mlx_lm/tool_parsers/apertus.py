# Copyright © 2026 Apple Inc.
"""
Modified from:
https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/apertus_tool_parser.py
"""

import json
from typing import Any

tool_call_start = "<|tools_prefix|>"
tool_call_end = "<|tools_suffix|>"


def parse_tool_call(text: str, tools: Any | None = None):
    # Apertus emits an array of single key objects mapping the function name to
    # its arguments, e.g.
    #   [{"get_weather": {"location": "London"}}, {"get_time": {}}]
    calls = json.loads(text)
    if not isinstance(calls, list):
        calls = [calls]

    tool_calls = []
    for call in calls:
        if not isinstance(call, dict) or not call:
            continue
        name, arguments = next(iter(call.items()))
        # A call with no arguments can come back as null, but clients expect an
        # object.
        if arguments is None:
            arguments = {}
        tool_calls.append(dict(name=name, arguments=arguments))

    if not tool_calls:
        raise ValueError(f"Could not parse tool call from: {text}")
    return tool_calls
