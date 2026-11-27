# Copyright © 2026 Apple Inc.

"""Tool call parser and response format for Inkling 2.

One assistant turn holds one or more messages, each with a header:

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Tokyo"}}<|end_message|>
    <|message_model|><|content_text|>answer<|end_message|><|content_model_end_sampling|>

The header text (including ``<|message_model|>`` and the tool name prefix before
``<|content_invoke_tool_json|>``) is dropped by the ``header`` state.
"""

import json
from typing import Any, Optional

tool_call_start = "<|content_invoke_tool_json|>"
tool_call_end = "<|end_message|>"

RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("<|content_thinking|>", "reasoning"),
            ("<|content_text|>", "normal"),
            ("<|content_invoke_tool_json|>", "tool"),
        ],
        "reasoning": [("<|end_message|>", "header")],
        "normal": [
            ("<|end_message|>", "header"),
            ("<|content_model_end_sampling|>", "normal"),
        ],
    },
}


def parse_tool_call(
    model_output: str, tools: Optional[Any] = None
) -> dict[str, Any]:
    raw = model_output.strip()
    if not raw:
        raise ValueError("Empty tool call.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in Inkling 2 tool call: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Inkling 2 tool call must be a JSON object.")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Inkling 2 tool call is missing a valid function name.")

    if "args" in data:
        arguments = data["args"]
    elif "arguments" in data:
        arguments = data["arguments"]
    else:
        arguments = {}

    if not isinstance(arguments, dict):
        raise ValueError("Inkling 2 tool call arguments must be a JSON object.")

    return {"name": name, "arguments": arguments}
