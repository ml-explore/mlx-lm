# Copyright © 2025 Apple Inc.

import json
from typing import Any, Optional

import regex as re

# FunctionGemma writes string arguments using <escape> delimiters.  Treat a
# complete literal as one token while matching braces so source code, YAML, and
# JSON inside an argument cannot terminate the surrounding function call.
_FUNCTION_GEMMA_STR = r"<escape>(?:(?!<escape>)[\s\S])*?<escape>"

_tool_call_regex = re.compile(
    r"call:([\w-]+)(\{(?:[^{}<]|<(?!escape>)|"
    + _FUNCTION_GEMMA_STR
    + r"|(?2))*\})",
    re.DOTALL,
)


def _function_gemma_args_to_json(text: str) -> str:
    """Convert FunctionGemma's compact arguments into valid JSON."""

    strings = []

    def _capture(match: re.Match) -> str:
        strings.append(match.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = re.sub(r"<escape>(.*?)<escape>", _capture, text, flags=re.DOTALL)
    text = re.sub(
        r"(?<=[{,])([A-Za-z_][A-Za-z0-9_-]*):",
        r'"\1":',
        text,
    )
    for index, value in enumerate(strings):
        text = text.replace(f"\x00{index}\x00", json.dumps(value))
    return text


def _parse_single(match: re.Match) -> dict:
    return {
        "name": match.group(1),
        "arguments": json.loads(_function_gemma_args_to_json(match.group(2))),
    }


def parse_tool_call(text: str, _: Optional[Any] = None):
    matches = list(_tool_call_regex.finditer(text))
    if not matches:
        raise ValueError("No complete function call provided.")
    if len(matches) == 1:
        return _parse_single(matches[0])
    return [_parse_single(match) for match in matches]


tool_call_start = "<start_function_call>"
tool_call_end = "<end_function_call>"
