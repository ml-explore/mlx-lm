import json
from typing import Any

import regex as re

tool_call_start: str = "<minimax:tool_call>"
tool_call_end: str = "</minimax:tool_call>"

_invoke_complete_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)

_parameter_complete_regex = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)


def _extract_name(name_str: str) -> str:
    """Extract name from quoted string."""
    name_str = name_str.strip()
    if (
        name_str.startswith('"')
        and name_str.endswith('"')
        or name_str.startswith("'")
        and name_str.endswith("'")
    ):
        return name_str[1:-1]
    return name_str


def _extract_types_from_schema(schema: Any) -> list[str]:
    """
    Extract all possible types from a JSON schema definition.
    Handles anyOf, oneOf, allOf, type arrays, and enum fields.

    Args:
        schema: The JSON schema definition for a parameter

    Returns:
        List of type strings (e.g., ["string", "integer", "null"])
    """
    if schema is None:
        return ["string"]

    if not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    # Handle direct "type" field
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)

    # Handle enum - infer types from enum values
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    # Handle anyOf, oneOf, allOf - recursively extract types
    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                extracted = _extract_types_from_schema(choice)
                types.update(extracted)

    # If no types found, default to string
    if not types:
        return ["string"]

    return list(types)


def _convert_param_value_with_types(value: str, param_types: list[str]) -> Any:
    if value.lower() == "null":
        return None

    # Normalize types
    normalized_types = [t.lower() for t in param_types]

    # Try null first if it's in the list
    if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
        return None

    # Try each type in order of preference (most specific first, string as fallback)
    # Priority: integer > number > boolean > object > array > string
    type_priority = [
        "integer",
        "int",
        "number",
        "float",
        "boolean",
        "bool",
        "object",
        "array",
        "string",
        "str",
        "text",
    ]

    for param_type in type_priority:
        if param_type not in normalized_types:
            continue

        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                continue
        elif param_type in ["boolean", "bool"]:
            lower_val = value.lower().strip()
            if lower_val in ["true", "1", "yes", "on"]:
                return True
            elif lower_val in ["false", "0", "no", "off"]:
                return False
            continue
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                continue

    # Fallback: try JSON parse, then return as string
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _get_param_types_from_config(param_name: str, param_config: dict) -> list[str]:
    if param_name not in param_config:
        return ["string"]
    param_schema = param_config[param_name]
    return _extract_types_from_schema(param_schema)


def parse_tool_call(text: str, tools: list | None = None):
    invoke_match = _invoke_complete_regex.findall(text)
    if not invoke_match:
        raise ValueError("No tool call found")
    invoke_text = invoke_match[0]

    name_match = re.search(r"^([^>]+)", invoke_text)
    if not name_match:
        return None

    function_name = _extract_name(name_match.group(1))

    # Get parameter configuration
    param_config = {}
    if tools:
        for tool in tools:
            if func := tool.get("function", False):
                if func["name"] != function_name:
                    continue
                if params := func.get("parameters", False):
                    param_config = params.get("properties", {})

    # Extract parameters
    param_dict = {}
    for match in _parameter_complete_regex.findall(invoke_text):
        param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
        if param_match:
            param_name = _extract_name(param_match.group(1))
            param_value = param_match.group(2).strip()
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_type = _get_param_types_from_config(param_name, param_config)

            param_dict[param_name] = _convert_param_value_with_types(
                param_value, param_type
            )

    return dict(name=function_name, arguments=param_dict)
