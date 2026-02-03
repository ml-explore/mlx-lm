# Copyright © 2025 Apple Inc.

"""
Modified from:
https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/glm4_moe_tool_parser.py
"""

import ast
import json
import shlex
from typing import Any

import regex as re

_func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
_func_arg_regex = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"


def _get_string_arg_names(tool_name: str, tools: list[Any] | None) -> set[str]:
    if tools is None:
        return set()
    for tool in tools:
        func = tool.get("function")
        if not func or func.get("name") != tool_name:
            continue
        params = func.get("parameters") or {}
        properties = params.get("properties") or {}
        return {
            name
            for name, schema in properties.items()
            if schema.get("type") == "string"
        }
    return set()


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


# Normalize argument values based on tool schema types.
def _normalize_arguments(
    func_name: str,
    arguments: dict[str, Any],
    tools: list[Any] | None,
    string_args: set[str] | None = None,
) -> dict[str, Any]:
    if string_args is None:
        string_args = _get_string_arg_names(func_name, tools)
    normalized = {}
    for key, value in arguments.items():
        # Preserve declared string types; coerce others when values are strings.
        if key in string_args:
            normalized[key] = value if isinstance(value, str) else str(value)
            continue
        if isinstance(value, str):
            normalized[key] = _deserialize(value)
        else:
            normalized[key] = value
    return normalized


# Parse JSON tool call payloads used by some GLM outputs.
def _parse_json_tool_call(text: str, tools: list[Any] | None):
    try:
        parsed = json.loads(text.strip())
    except Exception:
        return None

    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            parsed = parsed[0]
    if not isinstance(parsed, dict):
        return None

    # Pull out name/arguments from known JSON shapes.
    name = None
    arguments = None
    if "name" in parsed and "arguments" in parsed:
        name = parsed.get("name")
        arguments = parsed.get("arguments")
    elif "function" in parsed and "arguments" in parsed:
        name = parsed.get("function")
        arguments = parsed.get("arguments")
    elif "tool" in parsed and isinstance(parsed.get("tool"), dict):
        tool = parsed["tool"]
        name = tool.get("name")
        arguments = tool.get("arguments")

    if isinstance(name, dict):
        arguments = arguments or name.get("arguments")
        name = name.get("name")

    if isinstance(arguments, str):
        arguments = _deserialize(arguments)

    string_args = _get_string_arg_names(name, tools) if isinstance(name, str) else None

    if isinstance(name, str) and arguments is None:
        return dict(name=name, arguments={})
    if isinstance(name, str) and isinstance(arguments, dict):
        return dict(
            name=name,
            arguments=_normalize_arguments(
                name, arguments, tools, string_args=string_args
            ),
        )

    return None


# Parse key=value tokens into an arguments dict.
def _parse_key_value_pairs(
    text: str,
    func_name: str,
    tools: list[Any] | None,
    string_args: set[str] | None = None,
) -> dict[str, Any] | None:
    if "=" not in text:
        return None
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens:
        return None

    if string_args is None:
        string_args = _get_string_arg_names(func_name, tools)

    arguments = {}
    for token in tokens:
        # Require key=value tokens to avoid mis-parsing freeform text.
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            return None
        if key in string_args:
            arguments[key] = value
        else:
            arguments[key] = _deserialize(value)
    return arguments


# Parse plain text tool calls like "name a=1 b=2" or "name {json}".
def _parse_plain_text_tool_call(text: str, tools: list[Any] | None):
    stripped = text.strip()
    if not stripped:
        return None

    # Handle "name\\n{...}" style payloads.
    if "\n" in stripped:
        first_line, rest = stripped.split("\n", 1)
        name = first_line.strip()
        rest = rest.strip()
        if name and rest:
            string_args = _get_string_arg_names(name, tools)
            arguments = _deserialize(rest)
            if isinstance(arguments, dict):
                return dict(
                    name=name,
                    arguments=_normalize_arguments(
                        name, arguments, tools, string_args=string_args
                    ),
                )

    # Split on whitespace to get name + arguments segment.
    name, _, rest = stripped.partition(" ")
    if not name:
        return None
    rest = rest.strip()
    if not rest:
        return dict(name=name, arguments={})

    string_args = _get_string_arg_names(name, tools)
    arguments = _deserialize(rest)
    if isinstance(arguments, dict):
        return dict(
            name=name,
            arguments=_normalize_arguments(
                name, arguments, tools, string_args=string_args
            ),
        )

    kv_arguments = _parse_key_value_pairs(rest, name, tools, string_args=string_args)
    if kv_arguments is not None:
        return dict(name=name, arguments=kv_arguments)

    return dict(name=name, arguments={"raw": rest})


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a GLM 4.7 tool call string into a name and arguments dict."""
    match = _func_name_regex.search(text)
    if not match:
        # Fallbacks for alternate formats seen in GLM tool calls.
        fallback = _parse_json_tool_call(text, tools)
        if fallback is not None:
            return fallback
        fallback = _parse_plain_text_tool_call(text, tools)
        if fallback is not None:
            return fallback
        return dict(name="unknown", arguments={"raw": text.strip()})

    func_name = match.group(1)
    string_args = _get_string_arg_names(func_name, tools)
    arg_dct = {}
    for match in _func_arg_regex.finditer(text):
        arg_key = match.group(1).strip()
        arg_val = match.group(2).strip()
        if arg_key not in string_args:
            arg_val = _deserialize(arg_val)
        arg_dct[arg_key] = arg_val
    return dict(name=func_name, arguments=arg_dct)
