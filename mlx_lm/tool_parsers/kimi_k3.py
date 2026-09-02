# Copyright © 2026 Apple Inc.
"""
Parser for the Kimi K3 XTML tool-call format.

The model emits tool calls inside a tools section:

    <|open|>tools<|sep|>
      <|open|>call tool="get_weather" index="1"<|sep|>
        <|open|>argument key="city" type="string"<|sep|>Tokyo<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

Arguments are typed per-key, or carried as a single raw JSON object block:

    <|open|>json type="object"<|sep|>{"city": "Tokyo"}<|close|>json<|sep|>
"""

import json
from typing import Any

import regex as re

tool_call_start = "<|open|>tools<|sep|>"
tool_call_end = "<|close|>tools<|sep|>"

_call_regex = re.compile(
    r"<\|open\|>call(?P<attrs>(?:(?!<\|sep\|>).)*)<\|sep\|>"
    r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
    re.DOTALL,
)
_argument_regex = re.compile(
    r"<\|open\|>argument(?P<attrs>(?:(?!<\|sep\|>).)*)<\|sep\|>"
    r"(?P<value>.*?)<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)
_json_regex = re.compile(
    r"<\|open\|>json(?:(?!<\|sep\|>).)*<\|sep\|>"
    r"(?P<value>.*?)<\|close\|>json<\|sep\|>",
    re.DOTALL,
)
_attr_regex = re.compile(r'(\w+)="((?:[^"])*)"')


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(text: str) -> dict:
    return {k: _unescape_attr(v) for k, v in _attr_regex.findall(text)}


def _decode_value(value: str, value_type: str) -> Any:
    if value_type == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_single_call(attrs_text: str, body: str) -> dict:
    attrs = _parse_attrs(attrs_text)
    name = attrs.get("tool")
    if not name:
        raise ValueError("Kimi K3 tool call is missing the tool name attribute.")

    json_match = _json_regex.search(body)
    if json_match is not None:
        raw = json_match.group("value")
        arguments = json.loads(raw)
        if not isinstance(arguments, dict):
            raise ValueError("Kimi K3 tool call JSON block must be an object.")
    else:
        arguments = {}
        for arg in _argument_regex.finditer(body):
            arg_attrs = _parse_attrs(arg.group("attrs"))
            key = arg_attrs.get("key")
            if key is None:
                raise ValueError("Kimi K3 tool argument is missing its key.")
            arguments[key] = _decode_value(
                arg.group("value"), arg_attrs.get("type", "string")
            )

    return dict(name=name, arguments=arguments)


def parse_tool_call(text: str, tools: Any | None = None):
    calls = []
    for m in _call_regex.finditer(text):
        try:
            calls.append(_parse_single_call(m.group("attrs"), m.group("body")))
        except (ValueError, json.JSONDecodeError):
            continue
    if not calls:
        raise ValueError("No tool call found.")
    return calls
