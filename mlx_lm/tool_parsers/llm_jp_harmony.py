# Copyright © 2026 Apple Inc.

"""
LLM-jp-4.1 tool calls (gpt-oss format with a space after every special token):

    get_weather<|channel|> commentary <|constrain|>  json<|message|> {"city": "Tokyo"}
    get_weather <|constrain|>  json<|message|> {"city": "Tokyo"}
"""

import json
import re
from typing import Any, Optional

_NAME = re.compile(r"^\s*([^\s<]+)")
_BODY = "<|message|>"


def parse_tool_call(text: str, _: Optional[Any] = None) -> dict:
    match = _NAME.match(text)
    if not match:
        raise ValueError("No function name found.")
    body_at = text.rfind(_BODY)
    if body_at < 0:
        raise ValueError("No tool call arguments found.")
    body = text[body_at + len(_BODY) :]
    if body.startswith(" "):
        body = body[1:]
    arguments = json.loads(body) if body.strip() else {}
    return dict(name=match.group(1), arguments=arguments)


tool_call_start = " to=functions."
tool_call_end = "<|end|>"
