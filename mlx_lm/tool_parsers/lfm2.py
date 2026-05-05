# Copyright © 2026 Apple Inc.

"""
Tool parser for LiquidAI LFM2 / LFM2.5 models.

The model emits pythonic tool calls wrapped in special tokens, e.g.:

    <|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>

By the time `parse_tool_call` is invoked, the surrounding sentinel tokens
have already been stripped by the server's state machine, so this parser
sees only the inner ``[func(args), ...]`` payload.

Function names may be dotted (``grocery.orderIngredients``) and argument
values may contain nested lists / dicts, so we parse with ``ast`` rather
than regex. JSON-style ``true`` / ``false`` / ``null`` are tolerated as a
fallback for models that occasionally emit JSON literals.
"""

import ast
from typing import Any

import regex as re

tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"

_TOOL_CALL_REGEX = re.compile(r"\s*\[.*\]\s*$", re.DOTALL)

_JSON_LITERALS = [
    (re.compile(r"\btrue\b"), "True"),
    (re.compile(r"\bfalse\b"), "False"),
    (re.compile(r"\bnull\b"), "None"),
]


def _func_name(node: ast.AST) -> str:
    """Resolve a Name/Attribute chain into a dotted string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_func_name(node.value)}.{node.attr}"
    raise ValueError(f"Unsupported function name node: {type(node).__name__}")


def _literal_eval(value_node: ast.AST, source: str) -> Any:
    try:
        return ast.literal_eval(value_node)
    except (ValueError, SyntaxError):
        # Fallback: model emitted JSON literals (true/false/null) inside an
        # otherwise pythonic structure. Re-render the node and substitute.
        rendered = ast.unparse(value_node)
        for pattern, replacement in _JSON_LITERALS:
            rendered = pattern.sub(replacement, rendered)
        return ast.literal_eval(rendered)


def _parse_call(node: ast.AST, source: str) -> dict:
    if not isinstance(node, ast.Call):
        raise ValueError(f"Expected a function call, got {type(node).__name__}")
    if node.args:
        raise ValueError("LFM2 tool calls only accept keyword arguments")
    name = _func_name(node.func)
    arguments = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("`**kwargs`-style arguments are not supported")
        arguments[kw.arg] = _literal_eval(kw.value, source)
    return {"name": name, "arguments": arguments}


def _parse(source: str) -> ast.AST:
    try:
        return ast.parse(source, mode="eval")
    except SyntaxError:
        # Retry with JSON literals normalised. We do this on the raw source
        # only as a fallback because a blanket replacement could otherwise
        # corrupt strings containing the words true/false/null.
        normalised = source
        for pattern, replacement in _JSON_LITERALS:
            normalised = pattern.sub(replacement, normalised)
        return ast.parse(normalised, mode="eval")


def parse_tool_call(text: str, tools: Any | None = None):
    text = text.strip()
    if not text:
        raise ValueError("Empty tool call text")
    if not _TOOL_CALL_REGEX.match(text):
        raise ValueError(
            f"LFM2 tool call must be a list expression like `[fn(...)]`, got: {text!r}"
        )

    tree = _parse(text)
    body = tree.body
    if not isinstance(body, ast.List) or not body.elts:
        raise ValueError("Tool output must be a non-empty list of function calls")
    if not all(isinstance(elt, ast.Call) for elt in body.elts):
        raise ValueError("Every element of the tool list must be a function call")

    calls = [_parse_call(elt, text) for elt in body.elts]
    if len(calls) == 1:
        return calls[0]
    return calls
