# Copyright © 2026 Apple Inc.

"""
Tool parser for Poolside Laguna XML-like tool calls.

Format:
<tool_call>function-name
<arg_key>argument-key</arg_key>
<arg_value>value-of-argument-key</arg_value>
</tool_call>
"""

import ast
import json
from typing import Any

import regex as re

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"

_tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
_arg_pair_regex = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: list[Any] | None,
) -> bool:
    if tools is None:
        return False
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") != tool_name:
            continue
        params = func.get("parameters") or {}
        arg_type = params.get("properties", {}).get(arg_name, {}).get("type")
        return arg_type == "string"
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _parse_single_call(text: str, tools: list[Any] | None):
    text = text.strip()
    match = _func_name_regex.search(text)
    if not match:
        func_name = text.split("\n", 1)[0].strip()
        return dict(name=func_name, arguments={})

    func_name = match.group(1).strip()
    arguments = {}
    for match in _arg_pair_regex.finditer(text):
        arg_key = match.group(1).strip()
        arg_val = match.group(2).strip()
        if not _is_string_type(func_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arguments[arg_key] = arg_val
    return dict(name=func_name, arguments=arguments)


def parse_tool_call(text: str, tools: list[Any] | None = None):
    matches = _tool_call_regex.findall(text)
    if matches:
        calls = [_parse_single_call(match, tools) for match in matches]
        return calls[0] if len(calls) == 1 else calls

    return _parse_single_call(text, tools)
