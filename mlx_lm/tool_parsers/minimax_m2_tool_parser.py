# Copyright © 2025 Apple Inc.

"""
MiniMax M2 XML-based tool call parser.

This parser handles the XML format used by MiniMax M2 models for tool calls.
MiniMax M2 uses a structured XML format instead of JSON for tool invocations.

Expected format:
    <minimax:tool_call>
    <invoke name="function_name">
    <parameter name="arg1">value1</parameter>
    <parameter name="arg2">value2</parameter>
    </invoke>
    </minimax:tool_call>

Multiple tool calls can be made in a single block:
    <minimax:tool_call>
    <invoke name="func1">
    <parameter name="x">1</parameter>
    </invoke>
    <invoke name="func2">
    <parameter name="y">2</parameter>
    </invoke>
    </minimax:tool_call>

Parameter values support various types:
    - Strings: <parameter name="query">search term</parameter>
    - Numbers: <parameter name="count">42</parameter>
    - Booleans: <parameter name="enabled">true</parameter>
    - Arrays: <parameter name="tags">["a", "b", "c"]</parameter>
    - Objects: <parameter name="config">{"key": "value"}</parameter>

Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base import ExtractedToolCallInformation, ToolCall, ToolParser

logger = logging.getLogger(__name__)


class MinimaxM2ToolParser(ToolParser):
    """
    Parser for MiniMax M2 XML-formatted tool calls.

    This parser extracts tool calls from MiniMax M2's XML format and converts
    them to the standard ToolCall format compatible with OpenAI's API.
    """

    # Regex patterns for parsing XML structure
    # Pattern to match complete <invoke name="...">...</invoke> blocks
    INVOKE_PATTERN = re.compile(
        r'<invoke\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</invoke>',
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern to match <parameter name="...">...</parameter> elements
    PARAMETER_PATTERN = re.compile(
        r'<parameter\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</parameter>',
        re.DOTALL | re.IGNORECASE,
    )

    # Alternative pattern for self-closing parameters (rare but possible)
    PARAMETER_SELF_CLOSING_PATTERN = re.compile(
        r'<parameter\s+name\s*=\s*["\']([^"\']+)["\']\s*'
        r'value\s*=\s*["\']([^"\']*)["\']'
        r'\s*/?>',
        re.IGNORECASE,
    )

    @property
    def tool_call_start(self) -> str:
        return "<minimax:tool_call>"

    @property
    def tool_call_end(self) -> str:
        return "</minimax:tool_call>"

    def parse_tool_calls(
        self, tool_call_text: str
    ) -> ExtractedToolCallInformation:
        """
        Parse MiniMax M2 XML-formatted tool calls.

        Args:
            tool_call_text: XML string containing tool call information
                (content between <minimax:tool_call> tags)

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        tool_calls = []
        text = tool_call_text.strip()

        if not text:
            return ExtractedToolCallInformation(
                tool_calls=[],
                content="",
                tools_called=False,
            )

        # Find all <invoke> blocks
        for invoke_match in self.INVOKE_PATTERN.finditer(text):
            function_name = invoke_match.group(1).strip()
            invoke_content = invoke_match.group(2)

            # Parse parameters from the invoke block
            arguments = self._parse_parameters(invoke_content)

            tool_call = ToolCall(
                name=function_name,
                arguments=arguments,
                id=None,
            )
            tool_calls.append(tool_call)

        # If no invoke blocks found, try alternative parsing
        if not tool_calls:
            tool_calls = self._try_alternative_parsing(text)

        return ExtractedToolCallInformation(
            tool_calls=tool_calls,
            content="",
            tools_called=len(tool_calls) > 0,
        )

    def _parse_parameters(self, invoke_content: str) -> Dict[str, Any]:
        """
        Parse parameter elements from invoke block content.

        Args:
            invoke_content: Content inside an <invoke> block

        Returns:
            Dictionary of parameter names to converted values
        """
        arguments = {}

        # Match standard <parameter name="...">value</parameter> format
        for param_match in self.PARAMETER_PATTERN.finditer(invoke_content):
            param_name = param_match.group(1).strip()
            param_value_str = param_match.group(2).strip()

            # Convert the string value to appropriate type
            arguments[param_name] = self._convert_param_value(param_value_str)

        # Also check for self-closing parameter format
        for param_match in self.PARAMETER_SELF_CLOSING_PATTERN.finditer(
            invoke_content
        ):
            param_name = param_match.group(1).strip()
            param_value_str = param_match.group(2).strip()

            if param_name not in arguments:  # Don't override existing
                arguments[param_name] = self._convert_param_value(param_value_str)

        return arguments

    def _convert_param_value(self, value_str: str) -> Any:
        """
        Convert a parameter value string to its appropriate Python type.

        MiniMax M2 supports various value types:
        - Strings (plain text or quoted)
        - Numbers (integers and floats)
        - Booleans (true/false)
        - JSON arrays
        - JSON objects

        Args:
            value_str: The string value from the parameter element

        Returns:
            Converted Python value
        """
        return self.convert_value(value_str)

    def _try_alternative_parsing(self, text: str) -> List[ToolCall]:
        """
        Try alternative parsing strategies for malformed XML.

        This handles cases where the XML might be slightly malformed
        or use non-standard formatting.

        Args:
            text: The tool call text that didn't match standard patterns

        Returns:
            List of parsed ToolCall objects (may be empty)
        """
        tool_calls = []

        # Try to find function name using looser pattern
        loose_invoke_pattern = re.compile(
            r'<invoke[^>]*name\s*=\s*["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</invoke>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in loose_invoke_pattern.finditer(text):
            function_name = match.group(1).strip()
            content = match.group(2)

            arguments = self._parse_parameters(content)

            # If no parameters found with standard pattern, try looser matching
            if not arguments:
                arguments = self._parse_parameters_loose(content)

            if function_name:
                tool_calls.append(
                    ToolCall(
                        name=function_name,
                        arguments=arguments,
                        id=None,
                    )
                )

        return tool_calls

    def _parse_parameters_loose(self, content: str) -> Dict[str, Any]:
        """
        Parse parameters with a looser pattern for malformed XML.

        Args:
            content: Content that may contain parameters in non-standard format

        Returns:
            Dictionary of parameter names to values
        """
        arguments = {}

        # Very loose pattern that might catch malformed parameters
        loose_param_pattern = re.compile(
            r'<parameter[^>]*name\s*=\s*["\']?([^"\'>\s]+)["\']?[^>]*>'
            r'(.*?)'
            r'</parameter>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in loose_param_pattern.finditer(content):
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()
            arguments[param_name] = self._convert_param_value(param_value)

        return arguments


class MinimaxM2StreamingState:
    """
    State tracker for streaming MiniMax M2 tool call parsing.

    This class maintains state while incrementally parsing tool calls
    during streaming generation. It tracks partial XML elements and
    accumulates content until complete tool calls can be extracted.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all streaming state."""
        self.accumulated_text: str = ""
        self.in_tool_call: bool = False
        self.in_invoke: bool = False
        self.in_parameter: bool = False
        self.current_function_name: Optional[str] = None
        self.current_param_name: Optional[str] = None
        self.current_param_value: str = ""
        self.accumulated_params: Dict[str, Any] = {}
        self.pending_tool_calls: List[ToolCall] = []

    def add_token(self, token_text: str) -> Optional[ToolCall]:
        """
        Process a new token and potentially return a complete tool call.

        Args:
            token_text: The text of the newly generated token

        Returns:
            A complete ToolCall if one was finished, None otherwise
        """
        self.accumulated_text += token_text

        # Check for tool call boundaries
        if "<minimax:tool_call>" in self.accumulated_text and not self.in_tool_call:
            self.in_tool_call = True
            idx = self.accumulated_text.find("<minimax:tool_call>")
            self.accumulated_text = self.accumulated_text[
                idx + len("<minimax:tool_call>"):
            ]

        if not self.in_tool_call:
            return None

        # Check for complete invoke blocks
        completed_call = self._try_extract_complete_invoke()

        # Check for end of tool call block
        if "</minimax:tool_call>" in self.accumulated_text:
            self.in_tool_call = False
            idx = self.accumulated_text.find("</minimax:tool_call>")
            self.accumulated_text = self.accumulated_text[
                idx + len("</minimax:tool_call>"):
            ]

        return completed_call

    def _try_extract_complete_invoke(self) -> Optional[ToolCall]:
        """
        Try to extract a complete invoke block from accumulated text.

        Returns:
            ToolCall if a complete invoke was found, None otherwise
        """
        parser = MinimaxM2ToolParser()
        match = parser.INVOKE_PATTERN.search(self.accumulated_text)

        if match:
            function_name = match.group(1).strip()
            invoke_content = match.group(2)
            arguments = parser._parse_parameters(invoke_content)

            # Remove the matched content
            self.accumulated_text = self.accumulated_text[match.end():]

            return ToolCall(
                name=function_name,
                arguments=arguments,
                id=None,
            )

        return None

    def get_pending_calls(self) -> List[ToolCall]:
        """
        Get and clear any pending complete tool calls.

        Returns:
            List of complete ToolCall objects
        """
        calls = self.pending_tool_calls
        self.pending_tool_calls = []
        return calls
