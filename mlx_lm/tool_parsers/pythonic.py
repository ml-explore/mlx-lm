# Copyright © 2026 Apple Inc.

import ast
from typing import Any, Dict, List

import regex as re

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


# The block is a LIST of calls: [f1(a=1), f2(b="x")]. Match each call
# individually (quote-aware, so ')' inside a string arg doesn't terminate
# the call) — a single non-greedy search over the whole block merges the
# args of call 1..n together and silently drops calls 2..n.
_tool_block_regex = re.compile(r"\[(.*)\]", re.DOTALL)
_single_call_regex = re.compile(
    r"(\w+)\(((?:[^()\"']|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')*)\)",
    re.DOTALL,
)
_tool_args_regex = re.compile(r'(\w+)=(?:"([^"]*)"|([^,]+))(?:,\s*|$)', re.DOTALL)


def parse_tool_call(text: str, tools: Any | None = None):
    block = _tool_block_regex.search(text)
    if not block:
        raise ValueError("No function provided.")

    calls = []
    for match in _single_call_regex.finditer(block.group(1)):
        func_name = match.group(1)
        args_str = match.group(2)

        arguments = {}
        if args_str:
            for pair in _tool_args_regex.findall(args_str):
                key = pair[0].strip()
                # pair[1] is quoted value, pair[2] is unquoted value
                value = pair[1] if pair[1] else pair[2].strip()

                # Try to parse the value using ast.literal_eval
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If parsing fails, keep as string
                    pass

                arguments[key] = value
        calls.append(dict(name=func_name, arguments=arguments))

    if not calls:
        raise ValueError("No function provided.")
    return calls[0] if len(calls) == 1 else calls


tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
