# Copyright © 2026 Apple Inc.

import ast
from typing import Any

import regex as re

"""
Tool parser for Olmo 3 function call format.

Parses assistant responses containing tool calls in formats like:
<function_calls>
function_name(arg1="value1", arg2=2)
</function_calls>

Multiple tool calls are newline-separated within the tags. Argument values
are JSON literals (null, true, false) instead of Python literals.
"""


_tool_call_regex = re.compile(r"^\s*(\w+)\((.*)\)\s*$", re.MULTILINE)
_tool_args_regex = re.compile(r'(\w+)=(?:"([^"]*)"|([^,]+))(?:,\s*|$)', re.DOTALL)

_JSON_LITERALS = {"null": None, "true": True, "false": False}


def _coerce(value: str):
    value = value.strip()
    if value in _JSON_LITERALS:
        return _JSON_LITERALS[value]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_tool_call(text: str, tools: Any | None = None):
    calls = []
    for match in _tool_call_regex.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2)
        arguments = {}
        if args_str:
            for key, quoted, raw in _tool_args_regex.findall(args_str):
                arguments[key.strip()] = quoted if quoted else _coerce(raw)
        calls.append({"name": func_name, "arguments": arguments})

    if not calls:
        raise ValueError("No function provided.")

    return calls if len(calls) > 1 else calls[0]


tool_call_start = "<function_calls>"
tool_call_end = "</function_calls>"
