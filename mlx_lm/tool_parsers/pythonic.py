# Copyright © 2026 Apple Inc.

import ast
from typing import Any

import regex as re

from . import json_tools

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


_tool_call_regex = re.compile(r"\[([\w.]+)\((.*?)\)\]", re.DOTALL)
_tool_args_regex = re.compile(r'(\w+)=(?:"([^"]*)"|([^,]+))(?:,\s*|$)', re.DOTALL)


def _function_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _function_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return None


def _parse_json_tool_call(text):
    text = text.strip()
    if text.startswith("<tool_call>") and text.endswith("</tool_call>"):
        text = text[len("<tool_call>") : -len("</tool_call>")].strip()

    if not text.startswith("{"):
        return None

    parsed = json_tools.parse_tool_call(text)
    if not isinstance(parsed, dict) or "name" not in parsed or "arguments" not in parsed:
        return None
    return parsed


def _parse_pythonic_tool_call(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    parsed = ast.parse(text[start : end + 1], mode="eval").body
    if not isinstance(parsed, ast.List) or not parsed.elts:
        return None

    call = parsed.elts[0]
    if not isinstance(call, ast.Call):
        return None

    func_name = _function_name(call.func)
    if func_name is None:
        return None

    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        arguments[keyword.arg] = ast.literal_eval(keyword.value)

    return dict(name=func_name, arguments=arguments)


def parse_tool_call(text: str, tools: Any | None = None):
    for parser in (_parse_json_tool_call, _parse_pythonic_tool_call):
        try:
            parsed = parser(text)
        except (SyntaxError, ValueError):
            parsed = None
        if parsed is not None:
            return parsed

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
