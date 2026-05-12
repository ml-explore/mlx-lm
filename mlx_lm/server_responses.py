import base64
import time
import uuid


def error_response(message, error_type="invalid_request", code=None, param=None):
    error = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code or error_type,
    }
    return {"error": error}


def _content_part_text(part):
    part_type = part.get("type")
    if part_type in ("input_text", "output_text", "text"):
        return part.get("text", "")
    if part_type == "refusal":
        return part.get("refusal", "")
    if part_type == "input_image":
        return f"[input_image: {part.get('image_url', '')}]"
    if part_type == "input_file":
        return f"[input_file: {part.get('filename') or part.get('file_url') or ''}]"
    if part_type == "input_video":
        return f"[input_video: {part.get('video_url', '')}]"
    return ""


def content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_text(part) for part in content if isinstance(part, dict))
    return str(content)


def tool_output_to_text(output):
    if isinstance(output, str):
        return output
    return content_to_text(output)


def input_to_messages(input_items, instructions=None, stored_items=None):
    stored_items = stored_items or {}
    if input_items is None:
        input_items = []
    if isinstance(input_items, str):
        input_items = [{"type": "message", "role": "user", "content": input_items}]

    messages = []
    normalized = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
        normalized.append({"type": "message", "role": "system", "content": instructions})

    for item in input_items:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects")
        if item.get("type") == "item_reference":
            item_id = item.get("id")
            if item_id not in stored_items:
                raise ValueError(f"Referenced item '{item_id}' was not found")
            item = stored_items[item_id]

        item_type = item.get("type", "message")
        normalized.append(item)

        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": content_to_text(item.get("content"))})
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4()}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": call_id,
                            "function": {
                                "name": item["name"],
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": tool_output_to_text(item.get("output", "")),
                }
            )
        elif item_type in ("reasoning", "compaction"):
            continue
        else:
            raise ValueError(f"Unsupported Responses input item type: {item_type}")

    return messages, normalized


def chat_tools(tools, tool_choice=None):
    if not tools or tool_choice == "none":
        return None
    allowed_names = None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            allowed_names = {tool_choice.get("name")}
        elif tool_choice.get("type") == "allowed_tools":
            allowed_names = {t.get("name") for t in tool_choice.get("tools", []) if t.get("name")}
    selected = tools if allowed_names is None else [t for t in tools if t.get("name") in allowed_names]

    chat_tools = []
    for tool in selected:
        if "function" in tool:
            chat_tools.append(tool)
        elif tool.get("type") == "function":
            chat_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {},
                    },
                }
            )
    return chat_tools or None


def validate_tool_choice(tools, tool_choice):
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice not in ("auto", "none", "required"):
            raise ValueError(f"Unsupported tool_choice value '{tool_choice}'")
        return
    if not isinstance(tool_choice, dict):
        raise ValueError("tool_choice must be a string or object")

    choice_type = tool_choice.get("type")
    if choice_type not in ("function", "allowed_tools"):
        raise ValueError(f"Unsupported tool_choice type '{choice_type}'")
    tool_names = {t.get("name") for t in tools or [] if t.get("name")}
    if choice_type == "function":
        name = tool_choice.get("name")
        if name not in tool_names:
            raise ValueError(f"tool_choice references unknown function '{name}'")
    elif choice_type == "allowed_tools":
        for tool in tool_choice.get("tools", []):
            name = tool.get("name")
            if name not in tool_names:
                raise ValueError(f"allowed_tools references unknown function '{name}'")


def function_call_ids(items):
    return {
        item.get("call_id")
        for item in items
        if isinstance(item, dict) and item.get("type") == "function_call"
    }


def tool_call_required(tool_choice):
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return True
        if tool_choice.get("type") == "allowed_tools":
            return tool_choice.get("mode") == "required"
    return False


def output_message(text, item_id=None, status="completed"):
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def tool_call_item(tool_call):
    function = tool_call.get("function", {})
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
        "name": function.get("name", ""),
        "arguments": function.get("arguments", "{}"),
    }


def usage(prompt_tokens, completion_tokens, cached_tokens=0, reasoning_tokens=0):
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens or 0},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens or 0},
    }


def response_object(handler, response_id, output, response_usage, status="completed", error=None, previous_response_id=None):
    body = handler.body
    return {
        "id": response_id,
        "object": "response",
        "created_at": handler.created,
        "completed_at": int(time.time()) if status in ("completed", "failed", "incomplete") else None,
        "status": status,
        "incomplete_details": None,
        "model": handler.requested_model,
        "previous_response_id": previous_response_id,
        "instructions": body.get("instructions"),
        "output": output,
        "error": error,
        "tools": body.get("tools") or [],
        "tool_choice": body.get("tool_choice", "auto"),
        "truncation": body.get("truncation", "disabled"),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "text": body.get("text") or {"format": {"type": "text"}},
        "top_p": handler.top_p,
        "presence_penalty": handler.presence_penalty,
        "frequency_penalty": handler.frequency_penalty,
        "top_logprobs": max(handler.top_logprobs, 0),
        "temperature": handler.temperature,
        "reasoning": body.get("reasoning") or {"effort": None, "summary": None},
        "usage": response_usage,
        "max_output_tokens": handler.max_tokens,
        "max_tool_calls": body.get("max_tool_calls"),
        "store": body.get("store", True),
        "background": body.get("background", False),
        "service_tier": body.get("service_tier", "default"),
        "metadata": body.get("metadata") or {},
        "safety_identifier": body.get("safety_identifier"),
        "prompt_cache_key": body.get("prompt_cache_key"),
    }


def event(event_type, sequence_number, **kwargs):
    payload = {"type": event_type, "sequence_number": sequence_number}
    payload.update(kwargs)
    return payload


def stored_response(response, input_items):
    stored = {
        "response": response,
        "input": input_items,
        "output": response.get("output", []),
        "items": {},
    }
    for item in stored["input"] + stored["output"]:
        if isinstance(item, dict) and item.get("id"):
            stored["items"][item["id"]] = item
    return stored


def compact_response(body, created):
    try:
        messages, _ = input_to_messages(body.get("input", ""), instructions=body.get("instructions"))
        text = "\n".join(m.get("content") or "" for m in messages)
    except ValueError:
        text = content_to_text(body.get("input", ""))
    encrypted = base64.b64encode(text.encode()).decode()
    return {
        "id": f"cmpct_{uuid.uuid4().hex}",
        "object": "response.compaction",
        "output": [
            {
                "id": f"cmp_{uuid.uuid4().hex}",
                "type": "compaction",
                "encrypted_content": encrypted,
                "created_by": "mlx_lm",
            }
        ],
        "created_at": created,
        "usage": usage(0, 0),
    }
