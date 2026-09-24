# Copyright © 2026 Apple Inc.

"""
Tool-call parser for openbmb/MiniCPM5 family.

The model emits XML of the form:

    <function name="get_weather">
        <param name="city">Beijing</param>
        <param name="date">2024-06-27</param>
    </function>

Multi-line or special-character values may be wrapped in CDATA:

    <param name="text"><![CDATA[multi
    line]]></param>

Ported from SGLang's MiniCPM5Detector (PR #25600).
"""

import ast
import json
from typing import Any, Optional

import regex as re

tool_call_start: str = "<function"
tool_call_end: str = "</function>"

# Full <function name="..."> opening tag (when the segment includes outer tags,
# as it does for unit-test inputs).
_func_name_full_regex = re.compile(
    r"<function\s+name=[\"']([^\"']+)[\"'][^>]*>", re.DOTALL
)

# Bare leading `name="..."` (when the state machine has stripped the outer
# <function...> tag and the segment starts with the attribute body).
_func_name_bare_regex = re.compile(
    r"^\s*name=[\"']([^\"']+)[\"'][^>]*>", re.DOTALL
)

_param_regex = re.compile(
    r"<param\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</param>", re.DOTALL
)

# A <param> tag with no name= attribute invalidates the whole call.
_param_missing_name_regex = re.compile(r"<param(?![^>]*\bname=)[^>]*>", re.DOTALL)

_cdata_regex = re.compile(r"^<!\[CDATA\[(.*)\]\]>$", re.DOTALL)


def _coerce_value(value: str, want_type: Optional[str]) -> Any:
    """Coerce a raw param string into the type declared by the tool schema.

    Strings pass through. Other types try strict JSON, then Python-literal,
    then fall back to the raw string.
    """
    if want_type == "string":
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _schema_for(tools: Optional[list], func_name: str):
    """Return (param_types, allowed_props, required_props) for func_name."""
    if not tools:
        return {}, set(), set()
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        if not func or func.get("name") != func_name:
            continue
        params = func.get("parameters") or {}
        if not isinstance(params, dict):
            return {}, set(), set()
        props = params.get("properties") or {}
        if not isinstance(props, dict):
            return {}, set(), set()
        types = {
            k: (v.get("type") if isinstance(v, dict) else None)
            for k, v in props.items()
        }
        required = set(params.get("required") or [])
        return types, set(props.keys()), required
    return {}, set(), set()


def parse_tool_call(text: str, tools: Optional[list] = None):
    """Parse one MiniCPM5 XML tool call.

    The mlx-lm state machine emits one segment per `<function>...</function>`
    pair, so this function returns a single call dict rather than a list.

    Raises ValueError on malformed XML or schema-violating calls; the server
    layer (`ToolCallFormatter`) converts that into a logged warning and drops
    the call.
    """
    m = _func_name_full_regex.search(text) or _func_name_bare_regex.match(text)
    if not m:
        raise ValueError("No tool call found")
    func_name = m.group(1)

    if _param_missing_name_regex.search(text):
        raise ValueError(f"Tool call '{func_name}' has <param> without name= attribute")

    param_types, allowed_props, required_props = _schema_for(tools, func_name)

    arguments: dict = {}
    for pm in _param_regex.finditer(text):
        key = pm.group(1)
        if allowed_props and key not in allowed_props:
            raise ValueError(f"Tool call '{func_name}' uses unknown param '{key}'")
        if key in arguments:
            raise ValueError(f"Tool call '{func_name}' has duplicate param '{key}'")
        raw = pm.group(2).strip()
        cdata = _cdata_regex.match(raw)
        value = cdata.group(1) if cdata else raw
        arguments[key] = _coerce_value(value, param_types.get(key))

    missing = required_props - arguments.keys()
    if missing:
        raise ValueError(
            f"Tool call '{func_name}' missing required params: {sorted(missing)}"
        )

    return {"name": func_name, "arguments": arguments}
