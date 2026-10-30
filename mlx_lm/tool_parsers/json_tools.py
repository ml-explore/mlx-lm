# Copyright © 2025 Apple Inc.

import json

tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def _rebalance_json_brackets(text: str) -> str:
    out = []
    stack = []
    in_string = False
    escape = False
    closers = {"{": "}", "[": "]"}
    prev_popped = None
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            prev_popped = None
            out.append(ch)
        elif ch in closers:
            stack.append(closers[ch])
            prev_popped = None
            out.append(ch)
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                prev_popped = stack.pop()
                out.append(ch)
            elif prev_popped != ch and len(stack) >= 2 and stack[-2] == ch:
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] == stack[-1]:
                    out.append(stack.pop())
                    prev_popped = stack.pop()
                    out.append(prev_popped)
                    i = j
        else:
            if not ch.isspace():
                prev_popped = None
            out.append(ch)
        i += 1

    if in_string:
        out.append('"')
    while stack:
        out.append(stack.pop())
    return "".join(out)


def parse_tool_call(text, tools=None):
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return json.loads(_rebalance_json_brackets(stripped))
