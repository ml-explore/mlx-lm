# Copyright © 2026 Apple Inc.

"""Chat template of Muse Spark Open (chat format v1.3.3)."""

import json
from typing import Any, Dict, List, Optional

BOS = "<|begin_of_text|>"
START = "<|start|>"
MESSAGE = "<|message|>"
EOT = "<|eot|>"
EOM = "<|eom|>"

TOOL_INSTRUCTIONS = """In this environment you have access to a set of tools you can use to answer the user's question.

You can invoke a function by writing a "<atem:function_calls>" block like the following as part of your reply to the user:
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
...
</atem:invoke>
</atem:function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:
// Tool metadata"""

TOOL_EXAMPLE = """Here's an example of how to call a function in the tool set:
(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)

to=example_tool_name.example_function_name

<atem:function_calls>
<atem:invoke name="example_tool_name.example_function_name">
<atem:parameter name="example_parameter_1">value_1</atem:parameter>
<atem:parameter name="example_parameter_2">This is the value for the second parameter
that can span
"multiple" lines
</atem:parameter>
</atem:invoke>
</atem:function_calls>"""

REASONING_STRENGTH = {"minimal": 1, "low": 16, "medium": 64, "high": 256}

# Each message starts in the hidden "header". The messages to "user" and, in
# Muse Spark, to "commentary" are the answer.
RESPONSE_FORMAT: Dict[str, Any] = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("to=user" + MESSAGE, "normal"),
            ("to=commentary" + MESSAGE, "normal"),
            (MESSAGE, "reasoning"),
        ],
        "reasoning": [
            ("<atem:function_calls>", "tool"),
            (EOM + START + "assistant", "header"),
        ],
        "normal": [(EOM + START + "assistant", "header")],
    },
}


def _tojson(value: Any) -> str:
    """As Jinja's tojson filter, without sorted keys."""
    out = json.dumps(value)
    for c, e in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("'", "\\u0027"),
    ):
        out = out.replace(c, e)
    return out


def _text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for part in content:
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append(part["text"].strip())
        elif kind in ("image", "input_image", "image_url"):
            parts.append("<image>")
    return "".join(parts)


def _header(
    role: str, *, recipient: Optional[str] = None, tool_name: Optional[str] = None
) -> str:
    to = f" to={recipient}" if recipient else ""
    name = f" {tool_name}" if tool_name else ""
    return f"{START}{role}{to}{name}{MESSAGE}"


def _tool_call(tool_call: Dict[str, Any]) -> str:
    function = tool_call["function"]
    args = function["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    params = []
    for key, value in args.items():
        if value is True:
            value = "true"
        elif value is False:
            value = "false"
        elif value is None:
            value = "null"
        elif isinstance(value, (dict, list, tuple)):
            value = _tojson(value)
        params.append(f'<atem:parameter name="{key}">{value}</atem:parameter>')
    return (
        f'<atem:function_calls>\n<atem:invoke name="{function["name"]}">\n'
        + "".join(params)
        + "\n</atem:invoke>\n</atem:function_calls>"
    )


def _tool_sets(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group the tools by name space."""
    sets = {"": {"name": "", "description": "", "functions": {}}}
    for tool in tools:
        f = tool.get("function", tool)
        ns, _, name = f.get("name", "").partition(".")
        if not name:
            ns, name = "", ns
        sets.setdefault(ns, {"name": ns, "description": "", "functions": {}})
        sets[ns]["functions"][name] = {
            "name": name,
            "description": f.get("description", ""),
            "parameters": f.get("parameters"),
        }
    return sets


def _tool_set(sets: Dict[str, Dict[str, Any]]) -> str:
    lines = [TOOL_INSTRUCTIONS]
    for ts in sets.values():
        lines.append(
            f'{{"name": "{ts["name"]}", "description": {_tojson(ts["description"])}}}'
        )
    lines.append("// Function schemas")
    for ts in sets.values():
        for f in ts["functions"].values():
            name = f"{ts['name']}.{f['name']}" if ts["name"] else f["name"]
            lines.append(
                f'{{"name": "{name}", "description": {_tojson(f["description"])}, '
                f'"parameters": {_tojson(f["parameters"])}}}'
            )
    return "\n".join(lines) + "\n\n" + TOOL_EXAMPLE


def _valid_recipients(
    sets: Optional[Dict[str, Dict[str, Any]]], *, private: bool, public: bool
) -> str:
    parts = (['"self"'] if private else []) + (["None"] if public else [])
    for ns, ts in (sets or {}).items():
        if ns:
            parts.append(f'"{ns}.*"')
        else:
            first = next(iter(ts["functions"]), "")
            parts.append(f'"[func_name] (e.g., {first})"')
    parts.append('"user"')
    return f"# Valid recipients: {', '.join(parts)}."


def apply_chat_template(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    *,
    model_identity: Optional[str] = None,
    cutoff_date: Optional[str] = None,
    current_date: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    allow_private_thinking: bool = True,
    allow_public_thinking: bool = True,
    enable_valid_recipient_rendering: bool = True,
    **kwargs: Any,
) -> str:
    # A leading system message becomes the custom instructions.
    instructions = None
    if messages and messages[0]["role"] == "system":
        instructions = _text(messages[0])
        messages = messages[1:]
    sets = _tool_sets(tools) if tools else None
    items = [
        model_identity,
        f"Knowledge cutoff: {cutoff_date}." if cutoff_date is not None else None,
        f"Current date: {current_date}." if current_date is not None else None,
        instructions,
        _tool_set(sets) if sets else None,
        (
            f"Reasoning strength: {REASONING_STRENGTH[reasoning_effort]}."
            if reasoning_effort in REASONING_STRENGTH
            else None
        ),
        (
            _valid_recipients(
                sets, private=allow_private_thinking, public=allow_public_thinking
            )
            if enable_valid_recipient_rendering
            else None
        ),
    ]
    system = "\n\n".join(i for i in items if i).strip()

    out = BOS
    if system:
        out += _header("system") + system + EOT
    tool_names = {}
    for message in messages:
        role = message["role"]
        if role in ("system", "developer", "user"):
            out += _header(role) + _text(message) + EOT
        elif role == "assistant":
            reasoning = message.get("reasoning")
            if reasoning:
                out += _header(role, recipient="self") + reasoning + EOM
            if tool_calls := message.get("tool_calls"):
                for i, tool_call in enumerate(tool_calls):
                    name = tool_call["function"]["name"]
                    tool_names[tool_call.get("id")] = name
                    out += _header(role, recipient=name) + _tool_call(tool_call)
                    out += EOT if i == len(tool_calls) - 1 else EOM
            else:
                text = _text(message)
                if text or not reasoning:
                    out += _header(role, recipient="user") + text + EOT
        elif role == "tool":
            call_id = message.get("tool_call_id")
            out += _header(role, tool_name=tool_names.get(call_id, call_id))
            out += _text(message) + EOT
    if add_generation_prompt:
        out += START + "assistant"
    return out
