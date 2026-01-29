# Copyright © 2026 Apple Inc.
"""
Modified from:
https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/kimi_k2_tool_parser.py
"""

import ast
import json
from typing import Any

import regex as re

# kimi has a fixed function naming scheme, with a json formatted arg
#   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
_func_name_regex = re.compile(
    r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
)
_func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
_tool_call_split_regex = re.compile(
    r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
)

tool_call_start = "<|tool_calls_section_begin|>"
tool_call_end = "<|tool_calls_section_end|>"


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


def _parse_single_tool(text: str) -> dict:
    func_name_match = _func_name_regex.search(text)
    if func_name_match is None:
        raise ValueError("No tool call found.")
    tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
    func_name = func_name_match.group(2)  # e.g. "get_weather"

    func_args_match = _func_arg_regex.search(text)
    if func_args_match is None:
        raise ValueError("No tool call arguments found.")
    func_args = func_args_match.group(1)
    arg_dct = _deserialize(func_args)

    return dict(id=tool_call_id, name=func_name, arguments=arg_dct)


def parse_tool_call(text: str, tools: Any | None = None):
    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]
    else:
        return [_parse_single_tool(text)]
