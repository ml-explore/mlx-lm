# Copyright © 2026 Apple Inc.

"""
Tool parser for MiniCPM5 XML tool calls.

Modified from:
https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/minicpm5xml_tool_parser.py
"""

import ast
import json
import xml.etree.ElementTree as ET
from typing import Any

import regex as re

_FUNC_NAME_REGEX = re.compile(r"^\s*name=['\"]([^'\"]+)['\"][^>]*>")
_PARAM_REGEX = re.compile(
    r"<param\s+name=['\"]([^'\"]+)['\"]>([\s\S]*?)</param>", re.DOTALL
)
_FUNC_BLOCK_REGEX = re.compile(r"<function.*?</function>", re.DOTALL)

tool_call_start = "<function"
tool_call_end = "</function>"


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: list[Any] | None,
) -> bool:
    """Check if the argument type is string in the tool schema."""
    if tools is None:
        return False
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") == tool_name:
            params = func.get("parameters")
            if params is None:
                return False
            arg_type = params.get("properties", {}).get(arg_name, {}).get("type", None)
            return arg_type in {"string", "str"}
    return False


def _deserialize(value: str) -> Any:
    """Deserialize a value from string, trying JSON first then literal eval."""
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _parse_function_block(block: str, tools: list[Any] | None) -> dict | None:
    """Parse a single <function name="...">...</function> block.

    Returns {"name": ..., "arguments": {...}} or None on failure.
    """
    try:
        root = ET.fromstring(block)
        func_node = root if root.tag == "function" else None
        if func_node is None:
            return None
        func_name = (func_node.attrib.get("name") or "").strip()
        if not func_name:
            return None
        param_config = {}
        if tools:
            for tool in tools:
                f = tool.get("function", {})
                if f.get("name") == func_name:
                    params = f.get("parameters", {})
                    param_config = params.get("properties", {})
        arguments = {}
        for param in func_node.findall("param"):
            key = param.attrib.get("name")
            if not key:
                continue
            val_text = (param.text or "").strip()
            if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
                val_text = val_text[len("<![CDATA[") : -len("]]>")]
            if not _is_string_type(func_name, key, tools):
                val_text = _deserialize(val_text)
            arguments[key] = val_text
        return {"name": func_name, "arguments": arguments}
    except ET.ParseError:
        pass

    # Fallback: regex-based parse for malformed XML
    m = _FUNC_NAME_REGEX.search(block)
    if not m:
        return None
    func_name = m.group(1).strip()
    arguments = {}
    param_config = {}
    if tools:
        for tool in tools:
            f = tool.get("function", {})
            if f.get("name") == func_name:
                params = f.get("parameters", {})
                param_config = params.get("properties", {})
    for pm in _PARAM_REGEX.finditer(block):
        key = pm.group(1).strip()
        val_text = (pm.group(2) or "").strip()
        if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
            val_text = val_text[len("<![CDATA[") : -len("]]>")]
        if not _is_string_type(func_name, key, tools):
            val_text = _deserialize(val_text)
        arguments[key] = val_text
    return {"name": func_name, "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a MiniCPM5 XML tool call response.

    Accepts text containing <function name="..."> blocks and returns a
    list of tool call dicts, or a single dict if only one call is found.

    The text state machine strips the ``<function`` and ``</function>``
    markers, so this parser wraps them back internally.
    """
    # Reconstruct the full function block(s) since the text state machine
    # strips the start and end markers.
    text = text.strip()
    if text.startswith("<function"):
        # Markers are already present (e.g. direct parser call in tests)
        wrapped = text
    else:
        wrapped = f"<function {text}</function>"

    blocks = _FUNC_BLOCK_REGEX.findall(wrapped)
    if not blocks:
        # Try fallback: maybe the text is a single function block itself
        parsed = _parse_function_block(wrapped, tools)
        if parsed is not None:
            return parsed
        raise ValueError("No function call found in text")

    results = []
    for block in blocks:
        parsed = _parse_function_block(block, tools)
        if parsed is not None:
            results.append(parsed)

    if not results:
        raise ValueError("No valid function calls found")

    if len(results) == 1:
        return results[0]
    return results
