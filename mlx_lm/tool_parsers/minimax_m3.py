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
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)

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
        except ET.ParseError as e:
            # Do NOT silently fall back to {} — that reaches the client as
            # arguments: "{}", fails tool validation with no hint of why, and
            # lets any narration the model emitted alongside the call stand
            # as a phantom success. Surface an explicit error payload instead:
            # the client's validation error echoes it back to the model,
            # which can then retry with fixed args. Also log it so the server
            # has primary evidence of M3 emitting malformed argument XML.
            logger.warning(
                "minimax_m3: malformed argument XML for invoke %r (%s); "
                "raw body[:300]=%r", name, e, body[:300],
            )
            args = {
                "__parse_error__": (
                    f"MiniMax-M3 emitted malformed argument XML for tool "
                    f"'{name}' ({e}). Re-issue this tool call with each "
                    f"argument as a simple <key>value</key> tag. "
                    f"Raw snippet: {body[:200]!r}"
                )
            }
        if not isinstance(args, dict):
            if args == "":
                # Legitimate no-argument call, e.g. <invoke name="x"></invoke>
                # (empty body parses to an empty scalar, not a dict).
                args = {}
            else:
                logger.warning(
                    "minimax_m3: arguments for invoke %r parsed to %s, not "
                    "an object; body[:300]=%r",
                    name, type(args).__name__, body[:300],
                )
                args = {
                    "__parse_error__": (
                        f"Arguments for tool '{name}' parsed to "
                        f"{type(args).__name__} instead of an object. "
                        f"Re-issue the call with named <key>value</key> "
                        f"argument tags."
                    )
                }
        calls.append({"name": name, "arguments": args})

    if not calls:
        raise ValueError("No tool call found")
    if len(calls) == 1:
        return calls[0]
    return calls
