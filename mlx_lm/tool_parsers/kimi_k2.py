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
    r"^\s*(.+):\d+\s*<\|tool_call_argument_begin\|>", re.DOTALL
)
_func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)

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


def parse_tool_call(text: str, tools: Any | None = None):
    text = text.removeprefix("<|tool_call_begin|>").removesuffix("<|tool_call_end|>")
    func_name = _func_name_regex.search(text).group(1)
    # strip off the `functions.` prefix, if it exists.
    func_name = func_name[func_name.find(".") + 1 :]

    func_args = _func_arg_regex.search(text).group(1)
    arg_dct = _deserialize(func_args)

    return dict(name=func_name, arguments=arg_dct)
