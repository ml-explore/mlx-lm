# Copyright © 2025 Apple Inc.

import json


def parse_tool_text(text, tools=None):
    tool_call = json.loads(text.strip())
    return {
        "name": tool_call.get("name", None),
        "arguments": json.dumps(tool_call.get("arguments", "")),
    }
