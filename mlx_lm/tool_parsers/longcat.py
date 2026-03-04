# Copyright © 2026 Apple Inc.

import ast
import json
from typing import Any

import regex as re

_func_name_regex = re.compile(r"^(.*?)<longcat_arg_key>", re.DOTALL)
_func_arg_regex = re.compile(
    r"<longcat_arg_key>(.*?)</longcat_arg_key>(?:\\n|\s)*<longcat_arg_value>(.*?)</longcat_arg_value>",
    re.DOTALL,
)

tool_call_start = "<longcat_tool_call>"
tool_call_end = "</longcat_tool_call>"


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: list[Any] | None,
) -> bool:
    if tools is None:
        return False
    for tool in tools:
        func = tool["function"]
        if func["name"] == tool_name:
            params = func["parameters"]
            if params is None:
                return False
            arg_type = params.get("properties", {}).get(arg_name, {}).get("type", None)
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


def parse_tool_call(text: str, tools: list[Any] | None = None):
    text = text.strip()

    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    func_name = _func_name_regex.search(text).group(1).strip()
    pairs = _func_arg_regex.findall(text)
    arg_dct = {}
    for key, value in pairs:
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(func_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arg_dct[arg_key] = arg_val
    return dict(name=func_name, arguments=arg_dct)
