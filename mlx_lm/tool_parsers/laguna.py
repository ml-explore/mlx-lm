# Copyright © 2026 Apple Inc.

"""
Tool parser for Poolside Laguna.

Laguna emits the same <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>
format as GLM-4.7, so the glm47 parser handles it unchanged. Converted
checkpoints set tool_parser_type to "laguna", which needs this module to exist.
"""

from .glm47 import parse_tool_call, tool_call_end, tool_call_start

__all__ = ["parse_tool_call", "tool_call_start", "tool_call_end"]
