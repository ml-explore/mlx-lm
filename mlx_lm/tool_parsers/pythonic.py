# mlx_lm/tool_parsers/pythonic.py
# A custom tool parser for MLX-LM to handle Pythonic tool call formats
# as seen in formats like LFM2/2.5, Llama 3.2/4, with optional special tokens and brackets.

import regex as re
from typing import List, Dict, Any

"""
    Tool parser for Pythonic function call formats.

    Parses assistant responses containing tool calls in formats like:
    <|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


_tool_call_regex = re.compile(r'\[(\w+)\((.*?)\)\]', re.DOTALL)
_tool_args_regex = re.compile(r'(\w+)=(?:"([^"]*)"|([^,]+))(?:,\s*|$)', re.DOTALL)

def parse_tool_call(text: str, tools: Any | None = None):
    match = _tool_call_regex.findall(text)
    if not match:
        raise ValueError("No function provided.")

    func_name = match[0][0]
    args_str = match[0][1]

    arguments = {}
    if args_str:
        matches = _tool_args_regex.findall(args_str)
        for pair in matches:
            key = pair[0].strip()
            value = pair[1] if pair[1] else pair[2].strip()
            value = value.strip('"').strip("'")
            try:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
            except ValueError:
                pass  # keep as string
            arguments[key] = value

    return dict(name=func_name, arguments=arguments)

tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
