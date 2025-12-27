# Copyright © 2025 Apple Inc.

"""
Base classes for tool call parsing.

This module provides the abstract base class and data structures for parsing
tool calls from model output. Different models use different formats for
tool calls (e.g., JSON, XML), and this abstraction allows supporting multiple
formats through a common interface.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ToolCall:
    """
    Represents a single tool/function call extracted from model output.

    Attributes:
        name: The name of the function to call
        arguments: Dictionary of argument names to values
        id: Optional unique identifier for the tool call
    """

    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None

    def to_openai_format(self, index: int = 0) -> dict:
        """
        Convert to OpenAI API format.

        Args:
            index: The index of this tool call in the tool_calls array

        Returns:
            Dictionary in OpenAI's tool_calls format
        """
        import uuid

        return {
            "index": index,
            "id": self.id or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
                if isinstance(self.arguments, dict)
                else str(self.arguments),
            },
        }


@dataclass
class ExtractedToolCallInformation:
    """
    Contains the results of parsing tool calls from model output.

    Attributes:
        tool_calls: List of extracted tool calls
        content: Any text content outside of tool calls
        tools_called: Whether any tool calls were found
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    content: str = ""
    tools_called: bool = False


class ToolParser(ABC):
    """
    Abstract base class for tool call parsers.

    Each parser implementation handles a specific format for tool calls
    (e.g., JSON wrapped in <tool_call> tags, MiniMax XML format, etc.)

    Subclasses must implement:
        - tool_call_start: The token/string that marks the start of a tool call
        - tool_call_end: The token/string that marks the end of a tool call
        - parse_tool_calls: Method to extract tool calls from accumulated text
    """

    @property
    @abstractmethod
    def tool_call_start(self) -> str:
        """The token or string that marks the beginning of a tool call block."""
        pass

    @property
    @abstractmethod
    def tool_call_end(self) -> str:
        """The token or string that marks the end of a tool call block."""
        pass

    @abstractmethod
    def parse_tool_calls(
        self, tool_call_text: str
    ) -> ExtractedToolCallInformation:
        """
        Parse tool calls from the text between start and end markers.

        Args:
            tool_call_text: The text content between tool_call_start and
                tool_call_end markers (exclusive of the markers themselves)

        Returns:
            ExtractedToolCallInformation containing parsed tool calls
        """
        pass

    def extract_tool_calls_from_response(
        self, full_response: str
    ) -> Tuple[str, List[ToolCall]]:
        """
        Extract all tool calls from a complete model response.

        This method finds all tool call blocks in the response, parses them,
        and returns both the non-tool-call content and the extracted calls.

        Args:
            full_response: The complete model output text

        Returns:
            Tuple of (content_without_tool_calls, list_of_tool_calls)
        """
        tool_calls = []
        content_parts = []
        remaining = full_response

        while True:
            start_idx = remaining.find(self.tool_call_start)
            if start_idx == -1:
                content_parts.append(remaining)
                break

            # Add content before the tool call
            content_parts.append(remaining[:start_idx])

            # Find the end of the tool call
            end_idx = remaining.find(
                self.tool_call_end,
                start_idx + len(self.tool_call_start)
            )
            if end_idx == -1:
                # Incomplete tool call, treat as content
                content_parts.append(remaining[start_idx:])
                break

            # Extract and parse the tool call content
            tool_text = remaining[
                start_idx + len(self.tool_call_start):end_idx
            ]
            result = self.parse_tool_calls(tool_text)
            tool_calls.extend(result.tool_calls)

            # Continue after the tool call
            remaining = remaining[end_idx + len(self.tool_call_end):]

        content = "".join(content_parts).strip()
        return content, tool_calls

    @staticmethod
    def convert_value(value_str: str) -> Any:
        """
        Convert a string value to its appropriate Python type.

        Attempts to parse as JSON first, then falls back to type inference.

        Args:
            value_str: String representation of the value

        Returns:
            Converted value (bool, int, float, list, dict, or str)
        """
        value_str = value_str.strip()

        # Try JSON parsing first (handles arrays, objects, strings with quotes)
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass

        # Handle boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # Handle null/none
        if value_str.lower() in ("null", "none"):
            return None

        # Handle numbers
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str
