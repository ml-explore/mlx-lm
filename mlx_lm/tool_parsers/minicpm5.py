# Copyright © 2026 Apple Inc.

"""
Tool call parser for OpenBMB MiniCPM5.

The chat template asks the model for
``<function name="fn"><param name="p">value</param></function>`` with no
outer wrapper. Values containing ``<``, ``&`` or newlines are wrapped in a
CDATA block, and parallel calls are consecutive ``<function>`` blocks.
"""

from typing import Any, Optional

import regex as re

from .minimax_m2 import (
    _convert_param_value_with_types,
    _extract_name,
    _get_param_types_from_config,
)

tool_call_start = "<function name="
tool_call_end = "</function>"

# The state machine strips both markers, so the server hands the parser
# ``"fn"><param ...>...``; the leading ``<function name=`` is optional here
# so the same parser also accepts the full text.
_function_regex = re.compile(
    r"(?:<function\s+)?(?:name\s*=\s*)?"
    r"(?P<name>\"[^\"]*\"|'[^']*'|[^\s\"'<>]+)\s*>"
    r"(?P<body>.*?)(?:</function>|$)",
    re.DOTALL,
)
_param_regex = re.compile(
    r"<param\s+name\s*=\s*(?P<name>\"[^\"]*\"|'[^']*'|[^\s\"'<>]+)\s*>"
    r"(?P<value>.*?)</param>",
    re.DOTALL,
)
_cdata_regex = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.DOTALL)


def _param_value(raw: str) -> str:
    if (match := _cdata_regex.match(raw)) is not None:
        return match.group(1)
    return raw.strip()


def parse_tool_call(model_output: str, tools: Optional[Any] = None):
    function_matches = list(_function_regex.finditer(model_output))
    if not function_matches:
        raise ValueError("No function provided.")

    param_config_for = {}
    for tool in tools or []:
        if function := tool.get("function", False):
            if params := function.get("parameters", False):
                param_config_for[function["name"]] = params.get("properties", {})

    calls = []
    for function_match in function_matches:
        function_name = _extract_name(function_match.group("name"))
        param_config = param_config_for.get(function_name, {})
        arguments = {}
        for param_match in _param_regex.finditer(function_match.group("body")):
            param_name = _extract_name(param_match.group("name"))
            arguments[param_name] = _convert_param_value_with_types(
                _param_value(param_match.group("value")),
                _get_param_types_from_config(param_name, param_config),
            )
        calls.append(dict(name=function_name, arguments=arguments))

    if len(calls) == 1:
        return calls[0]
    return calls
