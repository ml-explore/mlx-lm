# Copyright © 2026 Apple Inc.

import ast
from typing import Any

import regex as re

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


ToolCall = dict[str, Any]

_tool_call_regex = re.compile(r"\[(?P<name>[\w.]+)\((?P<args>.*?)\)\]", re.DOTALL)
_tool_args_regex = re.compile(
    r"""(?P<key>\w+)="""
    r"""(?:"(?P<quoted>[^"]*)"|'(?P<quoted>[^']*)'|(?P<bare>[^,]+))"""
    r"""(?:,\s*|$)""",
    re.DOTALL,
)


class _JSONLiterals(ast.NodeTransformer):
    """Chat templates render nested containers as JSON, so those hold
    true/false/null where Python expects True/False/None."""

    _values = {"true": True, "false": False, "null": None}

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id not in self._values:
            return node
        return ast.copy_location(ast.Constant(self._values[node.id]), node)


def _function_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _function_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return None


def _parse_call(node: ast.expr) -> ToolCall | None:
    if not isinstance(node, ast.Call) or node.args:
        return None

    func_name = _function_name(node.func)
    if func_name is None:
        return None

    arguments = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        arguments[keyword.arg] = ast.literal_eval(_JSONLiterals().visit(keyword.value))

    return {"name": func_name, "arguments": arguments}


def _parse_pythonic_tool_call(text: str) -> ToolCall | list[ToolCall] | None:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None

    parsed = ast.parse(text[start : end + 1], mode="eval").body
    if not isinstance(parsed, ast.List) or not parsed.elts:
        return None

    # The payload can hold several calls, e.g. [a(x=1), b(y=2)].
    calls = [_parse_call(elt) for elt in parsed.elts]
    if any(call is None for call in calls):
        return None
    return calls[0] if len(calls) == 1 else calls


def parse_tool_call(text: str, tools: Any | None = None) -> ToolCall | list[ToolCall]:
    try:
        parsed = _parse_pythonic_tool_call(text)
    except (SyntaxError, ValueError):
        parsed = None
    if parsed is not None:
        return parsed

    # Fall back to the regex parser for anything that is not valid Python.
    match = _tool_call_regex.search(text)
    if not match:
        raise ValueError("No function provided.")

    func_name = match.group("name")
    args_str = match.group("args")

    arguments = {}
    for match in _tool_args_regex.finditer(args_str):
        bare = match.group("bare")
        key = match.group("key").strip()
        value = bare.strip() if bare is not None else match.group("quoted")

        # Try to parse the value using ast.literal_eval
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If parsing fails, keep as string
            pass

        arguments[key] = value

    return {"name": func_name, "arguments": arguments}


tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
