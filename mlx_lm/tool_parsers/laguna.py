# Copyright © 2026 Apple Inc.

from typing import Any

from . import glm47

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"


def _strip_names(tool_call):
    if isinstance(tool_call, list):
        return [_strip_names(tc) for tc in tool_call]
    if isinstance(tool_call, dict) and isinstance(tool_call.get("name"), str):
        tool_call = dict(tool_call)
        tool_call["name"] = tool_call["name"].strip()
    return tool_call


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse Laguna XML-like tool calls into a name and arguments dict."""
    match = glm47._func_name_regex.search(text)
    if not match:
        return _strip_names(glm47.parse_tool_call(text, tools))

    func_name = match.group(1).strip()
    string_args = glm47._get_string_arg_names(func_name, tools)
    arguments = {}
    for match in glm47._func_arg_regex.finditer(text):
        arg_key = match.group(1).strip()
        arg_val = match.group(2).strip()
        if arg_key not in string_args:
            arg_val = glm47._deserialize(arg_val)
        arguments[arg_key] = arg_val
    return dict(name=func_name, arguments=arguments)
