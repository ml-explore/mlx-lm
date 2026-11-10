# Copyright © 2026 Apple Inc.

"""Response format of gpt-oss harmony.

A turn is a header plus a channel body. The generation prompt already ends at
``<|start|>assistant``, so parsing starts in the header. Header text is
dropped.

    <|channel|>analysis<|message|>reasoning<|end|>
    <|channel|>final<|message|>answer<|return|>
    <|channel|>commentary<|message|>preamble<|end|>
    to=functions.<name> ... <|message|>{json}<|call|>

``to=functions.`` is the tool marker for both header orders: the recipient
may come before ``<|channel|>commentary`` or after it. ``<|call|>`` is also
an end-of-turn token, so a finished call is closed by ``finish`` when that
token is not in the text.
"""

RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("<|channel|>analysis<|message|>", "reasoning"),
            ("<|channel|>final<|message|>", "normal"),
            ("<|channel|>commentary<|message|>", "normal"),
            ("to=functions.", "tool"),
        ],
        "reasoning": [
            ("<|end|>", "header"),
            ("<|return|>", "stop"),
        ],
        "normal": [
            ("<|end|>", "header"),
            ("<|return|>", "stop"),
        ],
        "tool": [
            ("<|call|>", "header"),
            ("<|end|>", "header"),
            ("<|return|>", "stop"),
        ],
    },
}


def prepare_messages(messages: list) -> None:
    """Rewrite assistant turns into the shape the harmony template accepts.

    The template reads analysis from ``thinking`` and rejects a tool-call
    turn that sets both ``content`` and ``thinking``. Clients that follow
    the chat completion replay the ``reasoning`` field instead.
    """
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if not message.get("thinking"):
            for key in ("reasoning", "reasoning_content", "reasoning_text"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    message["thinking"] = value
                    break
        content = message.get("content")
        if (
            message.get("tool_calls")
            and message.get("thinking")
            and isinstance(content, str)
            and content
        ):
            message["content"] = ""
