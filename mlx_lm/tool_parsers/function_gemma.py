# Copyright © 2025 Apple Inc.

"""
Modified from:
    https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/functiongemma_tool_parser.py
"""

import json
from typing import Any, Optional

import regex as re

_tool_call_regex = re.compile(r"call:(\w+)\{(.*?)\}", re.DOTALL)
_arg_regex = re.compile(r"(\w+):<escape>(.*?)<escape>", re.DOTALL)


def parse_tool_call(text: str, _: Optional[Any] = None):
    match = _tool_call_regex.findall(text)
    if not match:
        raise ValueError("No function provided.")
    func_name = match[0][0]
    args_str = match[0][1]
    arguments = {}
    matches = _arg_regex.findall(args_str)
    for key, value in matches:
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value

    return {
        "name": func_name,
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


tool_call_start = "<start_function_call>"
tool_call_end = "<end_function_call>"
