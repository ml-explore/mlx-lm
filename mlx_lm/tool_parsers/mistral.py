# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

# Matches a single tool-call header of the form "name[ARGS]" and any trailing
# whitespace, positioned right before the JSON arguments object.
_tool_call_header_regex = re.compile(r"(\w+)\s*\[ARGS\]\s*", re.DOTALL)

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    # Mistral has no explicit tool-call end token, so the server accumulates the
    # whole tail of the generation into a single string. That string may contain
    # multiple concatenated calls (parallel tool calls, optionally separated by
    # "[TOOL_CALLS]") and/or trailing natural-language text. Parse each call by
    # locating a "name[ARGS]" header and decoding exactly one JSON object with
    # raw_decode, which correctly respects string/brace boundaries and ignores
    # any trailing data. Continue scanning after each decoded object.
    decoder = json.JSONDecoder()
    calls = []
    pos = 0
    while (match := _tool_call_header_regex.search(text, pos)) is not None:
        name = match.group(1)
        start = match.end()
        if start >= len(text) or text[start] != "{":
            pos = match.end()
            continue
        try:
            arguments, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = match.end()
            continue
        calls.append(dict(name=name, arguments=arguments))
        pos = end

    if not calls:
        raise ValueError(f"Could not parse tool call from: {text}")
    return calls[0] if len(calls) == 1 else calls
