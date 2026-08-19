# Copyright © 2025 Apple Inc.

"""DeepSeek-V4 DSML chat template.

DeepSeek-V4 encodes tool calls in DSML like DeepSeek-V3.2, but uses the
``<｜DSML｜tool_calls>`` block tag (V3.2 uses ``function_calls``), a distinct
tool-instruction preamble, and ``<tool_result>`` blocks for tool outputs. The
structural scaffolding (BOS, ``<｜User｜>``/``<｜Assistant｜>`` turns, ``<think>``
handling) and the parameter encoding are identical, so the leaf helpers are
reused from :mod:`mlx_lm.chat_templates.deepseek_v32`; only the V4-specific
templates and ``render_message`` control flow live here.

Wiring: set ``chat_template_type: "deepseek_dsml"`` in the model's
``tokenizer_config.json`` (paired with ``tool_parser_type: "deepseek_dsml"``).

Output is byte-exact with the reference ``encoding_dsv4.encode_messages`` for
single-turn system+tools+user prompts in both chat and thinking modes, and
supports assistant tool-call replay. Two multi-turn behaviors of the reference
are intentionally not ported (single-turn tool calling, the server's per-request
usage, does not reach them):

- Multi-tool-result merging/ordering (``merge_tool_messages`` /
  ``sort_tool_results_by_call_order``); tool messages render one
  ``<tool_result>`` block per message.
- Earlier-turn reasoning retention when tools are present (the reference forces
  ``drop_thinking=False`` in that case); here reasoning is rendered only for the
  assistant turn that follows the last user message.
"""

from typing import Any, Dict, List

from mlx_lm.chat_templates.deepseek_v32 import (
    bos_token,
    eos_token,
    thinking_start_token,
    thinking_end_token,
    dsml_token,
    system_msg_template,
    user_msg_template,
    assistant_msg_template,
    thinking_template,
    to_json,
    tools_from_openai_format,
    tool_calls_from_openai_format,
    encode_arguments_to_dsml,
    find_last_user_index,
    drop_thinking_messages,
)

# --- V4-specific templates (differ from V3.2: wording + the `tool_calls` tag) ---

TOOLS_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

tool_call_template = '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
tool_calls_template = "<{dsml_token}tool_calls>\n{tool_calls}\n</{dsml_token}tool_calls>"
tool_output_template = "<tool_result>{content}</tool_result>"


def _tool_call_with_str_args(tc: Dict[str, Any]) -> Dict[str, Any]:
    """``encode_arguments_to_dsml`` ``json.loads()``es ``tc["arguments"]``, i.e.
    it expects the OpenAI JSON *string* form. Some clients (e.g. agent loops
    replaying their own prior calls) send ``arguments`` already parsed as a
    dict; re-serialize those so multi-turn tool-call replay does not crash.
    """
    args = tc.get("arguments")
    if isinstance(args, str):
        return tc
    return {**tc, "arguments": to_json(args if args is not None else {})}


def render_tools(tools: List[Dict[str, Any]]) -> str:
    return TOOLS_SYSTEM_TEMPLATE.format(
        tool_schemas="\n".join(to_json(t) for t in tools),
        dsml_token=dsml_token,
    )


def render_message(index: int, messages: List[Dict[str, Any]], thinking_mode: str) -> str:
    assert thinking_mode in ("chat", "thinking"), f"Invalid thinking_mode `{thinking_mode}`"
    msg = messages[index]
    role = msg.get("role")
    content = msg.get("content")

    def turn_suffix() -> str:
        # Open a <think> block on the last user/developer turn and on the final
        # turn when generation follows it (e.g. after a tool result); every
        # other turn closes with </think>. Matches the reference for single
        # tool results; multi-result merging (see module note) is not ported.
        opens_generation = index in (find_last_user_index(messages), len(messages) - 1)
        if opens_generation and thinking_mode == "thinking":
            return thinking_start_token
        return thinking_end_token

    if role == "system":
        out = system_msg_template.format(content=content or "")
        if msg.get("tools"):
            out += "\n\n" + render_tools(tools_from_openai_format(msg["tools"]))
        return out

    if role == "user":
        return user_msg_template.format(content=content) + turn_suffix()

    if role == "tool":
        return user_msg_template.format(
            content=tool_output_template.format(content=content)
        ) + turn_suffix()

    if role == "assistant":
        reasoning = ""
        if thinking_mode == "thinking" and index > find_last_user_index(messages):
            reasoning = (
                thinking_template.format(reasoning_content=msg.get("reasoning_content") or "")
                + thinking_end_token
            )
        tool_calls_content = ""
        if msg.get("tool_calls"):
            rendered = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tc["name"],
                    arguments=encode_arguments_to_dsml(_tool_call_with_str_args(tc)),
                )
                for tc in tool_calls_from_openai_format(msg["tool_calls"])
            ]
            tool_calls_content = "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token, tool_calls="\n".join(rendered)
            )
        return assistant_msg_template.format(
            reasoning=reasoning, content=content or "", tool_calls=tool_calls_content
        )

    raise NotImplementedError(f"Unknown role: {role}")


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str = "thinking",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    tools: Any = None,
) -> str:
    full_messages = list(messages)

    # The reference encoder only recognizes tools attached to a (system) message
    # -- both for rendering them and for its "don't drop earlier reasoning when
    # tools are present" rule. The mlx-lm server, however, passes tools as a
    # top-level kwarg. Normalize by attaching the kwarg tools to a system message
    # (synthesizing an empty-content one if none exists) so a tools request
    # behaves identically whether tools arrive by kwarg or by message -- and is
    # never silently dropped for want of a system message to hang them on.
    if tools:
        sys_idx = next(
            (i for i, m in enumerate(full_messages) if m.get("role") == "system"), None
        )
        if sys_idx is None:
            full_messages.insert(0, {"role": "system", "content": "", "tools": tools})
        elif not full_messages[sys_idx].get("tools"):
            full_messages[sys_idx] = {**full_messages[sys_idx], "tools": tools}

    if thinking_mode == "thinking" and drop_thinking:
        full_messages = drop_thinking_messages(full_messages)

    prompt = bos_token if add_default_bos_token else ""
    for idx in range(len(full_messages)):
        prompt += render_message(idx, full_messages, thinking_mode=thinking_mode)
    return prompt


def apply_chat_template(
    messages,
    tools=None,
    continue_final_message=False,
    add_generation_prompt=False,
    enable_thinking=None,
    thinking_mode=None,
    **kwargs,
):
    # mlx-lm's TokenizerWrapper injects `enable_thinking` (the HF convention)
    # and the server passes `tools` as a top-level kwarg; a jinja template would
    # silently ignore any other kwargs, so a Python template must accept **kwargs
    # to stay a drop-in equivalent. `enable_thinking` is load-bearing for V4 (it
    # selects the thinking vs chat rendering), so map it onto our `thinking_mode`
    # unless the caller passed `thinking_mode` explicitly.
    if continue_final_message and add_generation_prompt:
        raise ValueError(
            "Only one of continue_final_message or add_generation_prompt can be True"
        )
    if thinking_mode is None:
        thinking_mode = "thinking" if (enable_thinking is None or enable_thinking) else "chat"
    encode_kwargs = {
        k: kwargs[k] for k in ("drop_thinking", "add_default_bos_token") if k in kwargs
    }
    out = encode_messages(
        messages, thinking_mode=thinking_mode, tools=tools, **encode_kwargs
    )
    if not add_generation_prompt and messages[-1]["role"] in ("user", "tool"):
        out = out.removesuffix(thinking_start_token).removesuffix(thinking_end_token)
        out = out.removesuffix("<｜Assistant｜>")
    if continue_final_message and messages[-1]["role"] == "assistant":
        out = out.removesuffix(eos_token)
    return out
