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


# Normalize argument values based on tool schema types.
def _normalize_arguments(
    func_name: str,
    arguments: dict[str, Any],
    tools: list[Any] | None,
) -> dict[str, Any]:
    normalized = {}
    for key, value in arguments.items():
        # Preserve declared string types; coerce others when values are strings.
        if _is_string_type(func_name, key, tools):
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

    if isinstance(name, str) and arguments is None:
        return dict(name=name, arguments={})
    if isinstance(name, str) and isinstance(arguments, dict):
        return dict(name=name, arguments=_normalize_arguments(name, arguments, tools))

    return None


# Parse key=value tokens into an arguments dict.
def _parse_key_value_pairs(
    text: str,
    func_name: str,
    tools: list[Any] | None,
) -> dict[str, Any] | None:
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens:
        return None

    arguments = {}
    for token in tokens:
        # Require key=value tokens to avoid mis-parsing freeform text.
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            return None
        if _is_string_type(func_name, key, tools):
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
            arguments = _deserialize(rest)
            if isinstance(arguments, dict):
                return dict(
                    name=name,
                    arguments=_normalize_arguments(name, arguments, tools),
                )

    # Split on whitespace to get name + arguments segment.
    name, _, rest = stripped.partition(" ")
    if not name:
        return None
    rest = rest.strip()
    if not rest:
        return dict(name=name, arguments={})

    arguments = _deserialize(rest)
    if isinstance(arguments, dict):
        return dict(
            name=name,
            arguments=_normalize_arguments(name, arguments, tools),
        )

    kv_arguments = _parse_key_value_pairs(rest, name, tools)
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
    pairs = _func_arg_regex.findall(text)
    arg_dct = {}
    for key, value in pairs:
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(func_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arg_dct[arg_key] = arg_val
    return dict(name=func_name, arguments=arg_dct)
