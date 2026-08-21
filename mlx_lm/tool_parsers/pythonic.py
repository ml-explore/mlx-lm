# Copyright © 2026 Apple Inc.

import ast
from typing import Any, Dict, List

import regex as re

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


_tool_call_regex = re.compile(r"\[(\w+)\((.*?)\)\]", re.DOTALL)
_tool_args_regex = re.compile(r'(\w+)=(?:"([^"]*)"|([^,]+))(?:,\s*|$)', re.DOTALL)


def parse_tool_call(text: str, tools: Any | None = None):
    match = _tool_call_regex.search(text)
    if not match:
        raise ValueError("No function provided.")

    func_name = match.group(1)
    args_str = match.group(2)

    arguments = {}
    if args_str:
        matches = _tool_args_regex.findall(args_str)
        for pair in matches:
            key = pair[0].strip()
            # pair[1] is quoted value, pair[2] is unquoted value
            value = pair[1] if pair[1] else pair[2].strip()

            # Try to parse the value using ast.literal_eval
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If parsing fails, keep as string
                pass

            arguments[key] = value

    return dict(name=func_name, arguments=arguments)


tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
