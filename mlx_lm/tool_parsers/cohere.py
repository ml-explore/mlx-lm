# Copyright © 2026

"""
Tool parser for Cohere North (cohere2_moe) action-block tool calls.

Format (from the North-Mini-Code chat template):
<|START_ACTION|>[
    {"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "SF"}}
]<|END_ACTION|>

The action block is a JSON array of calls; each has `tool_name` + `parameters`.
Thinking (`<|START_THINKING|>...<|END_THINKING|>`) and text
(`<|START_TEXT|>...<|END_TEXT|>`) live outside the action block and are handled
by the reasoning/detokenizer path, not here.
"""

import json
from typing import Any

import regex as re

tool_call_start = "<|START_ACTION|>"
tool_call_end = "<|END_ACTION|>"

_action_regex = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)


def _normalize(call: dict) -> dict:
    name = call.get("tool_name") or call.get("name") or call.get("function")
    args = call.get("parameters")
    if args is None:
        args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return dict(name=name, arguments=args or {})


def _parse_block(block: str):
    block = block.strip()
    try:
        data = json.loads(block)
    except Exception:
        # tolerate a bare single object without the array wrapper
        try:
            data = json.loads(f"[{block}]")
        except Exception:
            return None
    if isinstance(data, dict):
        data = [data]
    calls = [_normalize(c) for c in data if isinstance(c, dict)]
    calls = [c for c in calls if c["name"]]
    if not calls:
        return None
    return calls[0] if len(calls) == 1 else calls


def parse_tool_call(text: str, tools: list[Any] | None = None):
    match = _action_regex.search(text)
    if match:
        parsed = _parse_block(match.group(1))
        if parsed is not None:
            return parsed
    # markers already stripped by the streaming layer -> parse the remainder
    return _parse_block(text)
