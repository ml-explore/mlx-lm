# Copyright © 2025 Apple Inc.

import json

# The start marker intentionally omits the closing ">". Tool-call markers are matched as
# exact token-id sequences, and "<tool_call>" encodes to a run ending in a standalone ">"
# token. Many tokenizers merge that ">" with the following byte (e.g. "<tool_call>\n" -> a
# single ">\n" token), so the full-marker token run never appears in the generated stream
# and the call is never captured. Matching the stable "<tool_call" prefix avoids this;
# parse_tool_call() below tolerates the leftover ">" before the JSON.
tool_call_start = "<tool_call"

tool_call_end = "</tool_call>"


def parse_tool_call(text, tools=None):
    """Extract the first brace-balanced JSON object from the captured tool segment.

    Robust to a leading ">"/newline left by the prefix start marker and to a trailing
    "</tool_call>" if the end marker likewise merges. Output is identical to
    json.loads(text.strip()) for a clean JSON-only segment.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in tool call segment")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in tool call segment")
