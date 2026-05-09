import json

tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(text, tools=None):
    text = text.strip()
    tool_call, end = json.JSONDecoder().raw_decode(text)
    trailing = text[end:].strip()
    if trailing and set(trailing) != {"}"}:
        raise json.JSONDecodeError("Unexpected trailing data", text, end)
    return tool_call
