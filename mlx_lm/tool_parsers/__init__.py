# Copyright © 2025 Apple Inc.

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from .base import ToolParser, ToolCall, ExtractedToolCallInformation
from .minimax_m2_tool_parser import MinimaxM2ToolParser

TOOL_PARSERS = {
    "minimax_m2": MinimaxM2ToolParser,
}


def parse_tool_calls_to_openai_format(
    tool_text_list: List[str],
    tool_parser: Optional[ToolParser] = None,
) -> List[Dict[str, Any]]:
    """
    Parse tool call text strings and convert to OpenAI API format.

    Args:
        tool_text_list: List of raw tool call text strings
        tool_parser: Optional parser to use (falls back to JSON parsing if None)

    Returns:
        List of tool calls in OpenAI format
    """
    parsed_calls = []
    call_index = 0

    for tool_text in tool_text_list:
        if tool_parser is not None:
            result = tool_parser.parse_tool_calls(tool_text)
            for tc in result.tool_calls:
                parsed_calls.append(tc.to_openai_format(index=call_index))
                call_index += 1
        else:
            # Fallback JSON parsing for backward compatibility
            try:
                tool_call = json.loads(tool_text.strip())
                parsed_calls.append({
                    "index": call_index,
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name"),
                        "arguments": json.dumps(tool_call.get("arguments", "")),
                    },
                })
                call_index += 1
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse tool call: {tool_text}")

    return parsed_calls


def process_tool_call_arguments(message: Dict[str, Any]) -> None:
    """
    Process tool_calls in a message, converting arguments from JSON string to dict.

    This is necessary because OpenAI API returns arguments as a JSON string,
    but some chat templates expect it as a dict. Modifies message in place.

    Args:
        message: Message dict that may contain tool_calls
    """
    if "tool_calls" in message and message["tool_calls"]:
        for tool_call in message["tool_calls"]:
            if "function" in tool_call:
                args = tool_call["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        tool_call["function"]["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        pass


def get_tool_parser(parser_name: str) -> ToolParser:
    """
    Get a tool parser by name.

    Args:
        parser_name: Name of the parser (e.g., "json", "minimax_m2")

    Returns:
        ToolParser instance
    """
    if parser_name not in TOOL_PARSERS:
        raise ValueError(
            f"Unknown tool parser: {parser_name}. "
            f"Available parsers: {list(TOOL_PARSERS.keys())}"
        )

    return TOOL_PARSERS[parser_name]()


__all__ = [
    "ToolParser",
    "ToolCall",
    "ExtractedToolCallInformation",
    "MinimaxM2ToolParser",
    "get_tool_parser",
    "parse_tool_calls_to_openai_format",
    "process_tool_call_arguments",
    "TOOL_PARSERS",
]
