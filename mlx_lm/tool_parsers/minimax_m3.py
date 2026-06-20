"""Tool-call parser for MiniMax-M3.

M3's chat template emits tool calls wrapped in a namespaced XML dialect:

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="get_weather">
    ]<]minimax[>[<location>Paris]<]minimax[>[</location>
    ]<]minimax[>[<days>3]<]minimax[>[</days>
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

The ``]<]minimax[>[`` namespace token is prepended to every tag start/end.
Arguments are rendered by the template's ``to_xml`` macro:
  - mappings -> ``<key>value</key>`` children
  - iterables -> ``<item>value</item>`` children
  - primitives -> raw text (json-serialised for booleans)

This module strips the namespace tokens, isolates the ``<tool_call>``
payload, and walks each ``<invoke>`` block with ElementTree, mapping
``<item>...</item>`` children to lists and everything else to dicts.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

NAMESPACE_TOKEN: str = "]<]minimax[>["

# What omlx searches for in the model output to detect "tool call starts here".
# Must be exactly what the model emits, including the namespace token.
tool_call_start: str = NAMESPACE_TOKEN + "<tool_call>"
tool_call_end: str = NAMESPACE_TOKEN + "</tool_call>"

_invoke_pattern = re.compile(
    r'<invoke\s+name=(?:"([^"]+)"|\'([^\']+)\')\s*>(.*?)</invoke>',
    re.DOTALL,
)


def _strip_namespace(text: str) -> str:
    """Remove all M3 namespace prefix tokens, leaving clean XML."""
    return text.replace(NAMESPACE_TOKEN, "")


def _coerce_scalar(text: str) -> Any:
    """Try JSON parse for primitives, fall back to the raw string."""
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def _element_to_python(elem: ET.Element) -> Any:
    """Recursively convert an XML element into a Python value.

    Mirrors the M3 chat template's ``to_xml`` macro:
      - all <item> children -> list
      - other named children -> dict
      - leaf -> scalar (json-coerced if possible)
    """
    children = list(elem)
    if not children:
        return _coerce_scalar(elem.text)

    if all(c.tag == "item" for c in children):
        return [_element_to_python(c) for c in children]

    out: dict[str, Any] = {}
    for c in children:
        out[c.tag] = _element_to_python(c)
    return out


def parse_tool_call(text: str, tools: list | None = None):
    """Parse M3's tool-call XML format into structured tool_calls.

    Returns ``{"name": ..., "arguments": {...}}`` for a single call,
    or a list of such dicts for multiple. Raises ``ValueError`` if no
    ``<invoke>`` block is found (this signals omlx that the model
    didn't actually emit a tool call here).
    """
    cleaned = _strip_namespace(text)

    # If the <tool_call>...</tool_call> wrapper is present, narrow to it.
    if "<tool_call>" in cleaned and "</tool_call>" in cleaned:
        start = cleaned.find("<tool_call>") + len("<tool_call>")
        end = cleaned.find("</tool_call>", start)
        if end > start:
            cleaned = cleaned[start:end]

    calls = []
    for m in _invoke_pattern.finditer(cleaned):
        name = m.group(1) or m.group(2)
        body = m.group(3) or ""
        try:
            root = ET.fromstring(f"<args>{body}</args>")
            args = _element_to_python(root)
        except ET.ParseError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})

    if not calls:
        raise ValueError("No tool call found")
    if len(calls) == 1:
        return calls[0]
    return calls
