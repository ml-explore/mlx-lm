# Copyright © 2025 Apple Inc.

import json

tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def _rebalance_brackets(text: str) -> str:
    out = []
    stack = []
    in_string = False
    escape = False
    closers = {"{": "}", "[": "]"}

    for ch in text:
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            out.append(ch)
        elif ch in closers:
            stack.append(closers[ch])
            out.append(ch)
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
                out.append(ch)
        else:
            out.append(ch)

    if in_string:
        out.append('"')
    out.extend(reversed(stack))
    return "".join(out)


def parse_tool_call(text, tools=None):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_rebalance_brackets(text))
