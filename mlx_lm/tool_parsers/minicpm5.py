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

_FUNC_NAME_REGEX = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"][^>]*>")
_PARAM_REGEX = re.compile(
    r"<param\s+name=['\"]([^'\"]+)['\"]>([\s\S]*?)</param>", re.DOTALL
)
_FUNC_BLOCK_REGEX = re.compile(r"<function.*?</function>", re.DOTALL)

# SentencePiece/GPT-style decoders may emit U+0120 (Ġ) / U+010A (Ċ).
_TOKENIZER_SPACE = "\u0120"
_TOKENIZER_NEWLINE = "\u010a"

# JSON schema type aliases that map to a Python string.
_STRING_TYPES = {"string", "str"}

tool_call_start = "<function"
tool_call_end = "</function>"


class _ToolParseError(ValueError):
    """Internal: a tool call block violates the tool schema."""


def _normalize_model_output(text: str) -> str:
    """Normalize tokenizer artifacts before parsing.

    SentencePiece/GPT-style decoders may emit U+0120 (Ġ) / U+010A (Ċ) in place
    of spaces and newlines, and some model outputs collapse tag names and
    attributes (e.g. ``<functionname="foo">``). Normalizing here makes the
    XML and regex paths robust to these variants.
    """
    if (
        _TOKENIZER_SPACE not in text
        and _TOKENIZER_NEWLINE not in text
        and "<functionname=" not in text
        and "<paramname=" not in text
    ):
        return text

    normalized = text.replace(_TOKENIZER_SPACE, " ")
    normalized = normalized.replace(_TOKENIZER_NEWLINE, "\n")
    normalized = normalized.replace("<functionname=", "<function name=")
    normalized = normalized.replace("<paramname=", "<param name=")
    return normalized


def _build_tool_maps(tools: list[Any] | None):
    """Index the tool schemas for validation and type-aware conversion.

    Returns ``(tool_names, type_by_param, required_params)`` where
    ``type_by_param`` maps tool name -> {param name -> JSON schema type} and
    ``required_params`` maps tool name -> the set of required param names.
    """
    tool_names: set[str] = set()
    type_by_param: dict[str, dict[str, str]] = {}
    required_params: dict[str, set[str]] = {}

    for tool in tools or []:
        func = tool.get("function", {})
        name = func.get("name")
        if not name:
            continue
        tool_names.add(name)
        params = func.get("parameters") or {}
        props = params.get("properties") or {}
        type_by_param[name] = {
            key: (value.get("type") if isinstance(value, dict) else None)
            for key, value in props.items()
        }
        required = params.get("required") or []
        required_params[name] = set(required)

    return tool_names, type_by_param, required_params


def _parse_value(value: str) -> tuple[Any, bool]:
    """Parse a value as JSON first, then as a Python literal.

    Returns ``(parsed_value, ok)``; on failure ``ok`` is ``False`` and the raw
    string is returned unchanged.
    """
    try:
        return json.loads(value), True
    except Exception:
        pass
    try:
        return ast.literal_eval(value), True
    except Exception:
        pass
    return value, False


def _convert_argument(
    value_text: str,
    arg_type: str | None,
    *,
    has_schema: bool,
) -> Any:
    """Convert a parameter value based on its schema type.

    Rules:
    - ``string``/``str`` types (and fields with no declared type) are preserved
      as strings, so values that merely look like JSON never get coerced.
    - Known typed fields (integer, number, boolean, array, object, null) are
      deserialized and validated against the schema type; malformed values are
      rejected rather than silently kept as strings.
    - Without a schema there is no evidence about any field, so values are
      preserved as strings instead of assuming they are structured.
    """
    if not has_schema or arg_type is None or arg_type in _STRING_TYPES:
        return value_text

    parsed, ok = _parse_value(value_text)
    if not ok:
        raise _ToolParseError(f"value {value_text!r} is not a valid {arg_type}")

    if arg_type == "integer":
        if isinstance(parsed, float) and parsed.is_integer():
            return int(parsed)
        if isinstance(parsed, bool) or not isinstance(parsed, int):
            raise _ToolParseError(f"value {value_text!r} is not a valid integer")
        return parsed
    if arg_type == "number":
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise _ToolParseError(f"value {value_text!r} is not a valid number")
        return parsed
    if arg_type == "boolean":
        if not isinstance(parsed, bool):
            raise _ToolParseError(f"value {value_text!r} is not a valid boolean")
        return parsed
    if arg_type == "array":
        if not isinstance(parsed, list):
            raise _ToolParseError(f"value {value_text!r} is not a valid array")
        return parsed
    if arg_type == "object":
        if not isinstance(parsed, dict):
            raise _ToolParseError(f"value {value_text!r} is not a valid object")
        return parsed
    if arg_type == "null":
        return None
    return value_text


def _collect_arguments(
    func_name: str,
    raw_params: list[tuple[str | None, str]],
    tool_names: set[str],
    type_by_param: dict[str, dict[str, str]],
    required_params: dict[str, set[str]],
) -> dict[str, Any] | None:
    """Validate a function call against the tool schemas and build arguments.

    Returns the validated ``arguments`` dict, or ``None`` when the call is not
    a valid tool call: unknown function name, parameter without a name,
    duplicate parameter, missing required parameter, or a value that does not
    match its declared schema type. When a schema is available, parameters
    that are not declared in it are dropped (consistent with the vLLM
    reference parser) — they cannot be validated, so they are never emitted.
    """
    if not func_name:
        return None
    has_schema = bool(tool_names)
    if has_schema and func_name not in tool_names:
        return None

    allowed_props = type_by_param.get(func_name)

    arguments: dict[str, Any] = {}
    seen: set[str] = set()
    for key, value_text in raw_params:
        if not key:
            return None
        if has_schema and key not in allowed_props:
            # Unknown parameter: drop it rather than pass through unvalidated
            # (the vLLM reference silently ignores parameters absent from the
            # requested tool schema).
            continue
        if key in seen:
            return None
        seen.add(key)
        try:
            arguments[key] = _convert_argument(
                value_text,
                allowed_props.get(key) if has_schema else None,
                has_schema=has_schema,
            )
        except _ToolParseError:
            return None

    required = required_params.get(func_name)
    if has_schema and required and not required.issubset(arguments):
        return None

    return arguments


def _parse_function_block(
    block: str,
    tool_names: set[str],
    type_by_param: dict[str, dict[str, str]],
    required_params: dict[str, set[str]],
) -> dict | None:
    """Parse a single ``<function name="...">...</function>`` block.

    Returns ``{"name": ..., "arguments": {...}}`` or ``None`` when the block is
    not a valid tool call.
    """
    # Primary path: well-formed XML. With xml.etree.ElementTree CDATA sections
    # are parsed into ordinary element text (markers stripped), so param values
    # are taken verbatim here to preserve CDATA whitespace.
    try:
        root = ET.fromstring(block)
        func_node = root if root.tag == "function" else None
        if func_node is None:
            return None
        func_name = (func_node.attrib.get("name") or "").strip()
        if not func_name:
            return None
        raw_params = [
            (param.attrib.get("name"), param.text or "")
            for param in func_node.findall("param")
        ]
        arguments = _collect_arguments(
            func_name, raw_params, tool_names, type_by_param, required_params
        )
        if arguments is None:
            return None
        return {"name": func_name, "arguments": arguments}
    except ET.ParseError:
        pass

    # Fallback: regex-based parse for malformed XML (e.g. truncated output).
    m = _FUNC_NAME_REGEX.search(block)
    if not m:
        return None
    func_name = m.group(1).strip()
    raw_params = []
    for pm in _PARAM_REGEX.finditer(block):
        key = pm.group(1).strip()
        val_text = pm.group(2) or ""
        if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
            val_text = val_text[len("<![CDATA[") : -len("]]>")]
        else:
            val_text = val_text.strip()
        raw_params.append((key, val_text))
    arguments = _collect_arguments(
        func_name, raw_params, tool_names, type_by_param, required_params
    )
    if arguments is None:
        return None
    return {"name": func_name, "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a MiniCPM5 XML tool call response.

    The tokenizer text state machine strips the ``<function`` and
    ``</function>`` markers, so the OpenAI server and example scripts pass the
    payload between them, e.g.::

        name=\"get_weather\"><param name=\"city\">Tokyo</param>

    Complete ``<function...>...</function>`` blocks (unit tests, the
    vLLM-style extract path) are also accepted and normalized to the same
    form, so the parser never has to guess whether the markers were stripped.

    Tool calls are validated against ``tools`` when supplied: unknown function
    names are rejected, parameters absent from the schema are dropped,
    duplicate parameters and missing required parameters are rejected, and
    each value is converted according to its declared schema type (strings stay
    strings, typed fields are deserialized and validated). Without a schema,
    values are preserved as strings. Malformed or truncated calls raise
    ``ValueError`` so callers can skip them.

    Returns a single dict for one call, or a list of dicts for several calls.
    """
    text = _normalize_model_output(text or "").strip()

    tool_names, type_by_param, required_params = _build_tool_maps(tools)

    # The text state machine strips the outer markers, so reconstruct them.
    # Complete XML blocks (already carrying the markers) pass through
    # unchanged; either way parsing always sees <function ...</function>.
    if not text.startswith("<function"):
        text = f"<function {text}</function>"

    results = []
    for block in _FUNC_BLOCK_REGEX.findall(text):
        parsed = _parse_function_block(
            block, tool_names, type_by_param, required_params
        )
        if parsed is not None:
            results.append(parsed)

    if not results:
        # No complete block was found: the output may be a truncated tool call
        # (e.g. cut off at finish_reason="length"). Salvage what we can.
        parsed = _parse_function_block(text, tool_names, type_by_param, required_params)
        if parsed is not None:
            results.append(parsed)

    if not results:
        raise ValueError("No valid function calls found in text")
    if len(results) == 1:
        return results[0]
    return results
