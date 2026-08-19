# Tool parser for DeepSeek-V4's native DSML tool-call format. Deploy into
# venv: mlx_lm/tool_parsers/deepseek_dsml.py, set tokenizer_config
# "tool_parser_type": "deepseek_dsml", and use the official DSML chat template.
#
# Format (multiple invokes per tool_calls block => native parallel calls):
#   <｜DSML｜tool_calls>
#   <｜DSML｜invoke name="get_weather">
#   <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
#   <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
#   </｜DSML｜invoke>
#   ...
#   </｜DSML｜tool_calls>
#
# string="true"  -> value is a literal string
# string="false" -> value is JSON (numbers, booleans, arrays, objects)
#
# Start/end markers drop the trailing ">": mlx-lm matches markers by token-id
# sequence and BPE merges ">" with the next byte ("...calls>\n" -> one token),
# so the precomputed full marker never matches. "<｜DSML｜tool_calls" (the
# <,｜DSML｜,tool,_c,alls tokens) is stable; the parser tolerates the leftover ">".
import json

import regex as re

tool_call_start = "<｜DSML｜tool_calls"
tool_call_end = "</｜DSML｜tool_calls"

# Lenient invoke close (</｜DSML｜inv…>): DeepSeek-V4 quants systematically
# truncate </｜DSML｜invoke> to </｜DSML｜inv> (reproducible across 4/5/6-bit),
# which the strict close tag would otherwise reject.
_INVOKE = re.compile(r"<｜DSML｜invoke\s+name=(.*?)</｜DSML｜inv[^>]*>", re.DOTALL)
_PARAM = re.compile(
    r'<｜DSML｜parameter\s+name=(?P<name>"[^"]*"|\'[^\']*\'|[^\s>]+)'
    r'\s+string="(?P<is_str>true|false)"\s*>(?P<val>.*?)</｜DSML｜parameter>',
    re.DOTALL,
)
_NAME_BODY = re.compile(r'\s*(?P<name>"[^"]*"|\'[^\']*\'|[^\s>]+)\s*>(?P<body>.*)', re.DOTALL)


def _unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _strip_one_newline(v):
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def parse_tool_call(text, tools=None):
    calls = []
    for invoke in _INVOKE.findall(text):
        m = _NAME_BODY.match(invoke)
        if not m:
            continue
        name = _unquote(m.group("name"))
        args = {}
        for pm in _PARAM.finditer(m.group("body")):
            pname = _unquote(pm.group("name"))
            val = _strip_one_newline(pm.group("val"))
            if pm.group("is_str") == "true":
                args[pname] = val
            else:
                try:
                    args[pname] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    args[pname] = val
        # template fallback: whole arg blob passed as a single "arguments" param
        if set(args) == {"arguments"} and isinstance(args["arguments"], str):
            try:
                args = json.loads(args["arguments"])
            except (json.JSONDecodeError, ValueError):
                pass
        calls.append({"name": name, "arguments": args})
    if not calls:
        raise ValueError("no DSML tool call found")
    return calls[0] if len(calls) == 1 else calls
