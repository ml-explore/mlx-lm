# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

# Matches a "name[ARGS]" header, ending where the JSON arguments start.
_tool_call_header_regex = re.compile(r"([\w-]+)\s*\[ARGS\]\s*")

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    # Mistral has no tool-call end token, so the text can hold several calls.
    # raw_decode reads exactly one JSON value and reports where it ended.
    decoder = json.JSONDecoder()
    calls = []
    pos = 0
    while (match := _tool_call_header_regex.search(text, pos)) is not None:
        pos = match.end()
        try:
            arguments, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            continue
        calls.append(dict(name=match.group(1), arguments=arguments))

    if not calls:
        raise ValueError(f"Could not parse tool call from: {text}")
    return calls[0] if len(calls) == 1 else calls
