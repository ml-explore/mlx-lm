import base64
import hashlib
import json
import socket
import struct
import time
import uuid
from urllib.parse import parse_qs


def error_response(message, error_type="invalid_request", code=None, param=None):
    error = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code or error_type,
    }
    return {"error": error}


def is_responses_path(path):
    return path in ("/v1/responses", "/v1/responses/compact")


def write_json_error(handler, status_code, message, code=None, param=None):
    handler._set_completion_headers(status_code)
    handler.end_headers()
    if is_responses_path(handler.path):
        response = error_response(message, code=code, param=param)
    else:
        response = {"error": message}
    handler.wfile.write(json.dumps(response).encode())


def read_request_body(handler):
    content_length = handler.headers.get("Content-Length")
    if content_length is None:
        raise ValueError("Content-Length header is required")
    try:
        content_length = int(content_length)
    except ValueError as e:
        raise ValueError("Invalid Content-Length header") from e

    raw_body = handler.rfile.read(content_length)
    content_type = handler.headers.get("Content-Type", "application/json")
    if content_type.startswith("application/x-www-form-urlencoded"):
        parsed = parse_qs(raw_body.decode())
        body = {}
        for key, values in parsed.items():
            value = values[-1]
            try:
                body[key] = json.loads(value)
            except json.JSONDecodeError:
                body[key] = value
        return body

    try:
        body = json.loads(raw_body.decode())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {e}") from e
    if not isinstance(body, dict):
        raise ValueError("Request should be a JSON dictionary")
    return body


def load_generation_parameters(handler, body):
    handler.stream = body.get("stream", False)
    handler.stream_options = body.get("stream_options", None)
    handler.requested_model = body.get("model", "default_model") or "default_model"
    handler.requested_draft_model = body.get("draft_model", "default_model")
    handler.num_draft_tokens = body.get(
        "num_draft_tokens", handler.response_generator.cli_args.num_draft_tokens
    )
    handler.adapter = body.get("adapters", None)
    handler.max_tokens = body.get("max_output_tokens", None)
    if handler.max_tokens is None:
        handler.max_tokens = body.get("max_completion_tokens", None)
    if handler.max_tokens is None:
        handler.max_tokens = body.get(
            "max_tokens", handler.response_generator.cli_args.max_tokens
        )
    handler.temperature = body.get("temperature", handler.response_generator.cli_args.temp)
    handler.top_p = body.get("top_p", handler.response_generator.cli_args.top_p)
    handler.top_k = body.get("top_k", handler.response_generator.cli_args.top_k)
    handler.min_p = body.get("min_p", handler.response_generator.cli_args.min_p)
    handler.repetition_penalty = body.get("repetition_penalty", 0.0)
    handler.repetition_context_size = body.get("repetition_context_size", 20)
    handler.presence_penalty = body.get("presence_penalty", 0.0)
    handler.presence_context_size = body.get("presence_context_size", 20)
    handler.frequency_penalty = body.get("frequency_penalty", 0.0)
    handler.frequency_context_size = body.get("frequency_context_size", 20)
    handler.xtc_probability = body.get("xtc_probability", 0.0)
    handler.xtc_threshold = body.get("xtc_threshold", 0.0)
    handler.logit_bias = body.get("logit_bias", None)
    handler.logprobs = body.get("logprobs", False)
    handler.top_logprobs = body.get("top_logprobs", -1)
    handler.seed = body.get("seed", None)
    handler.chat_template_kwargs = body.get("chat_template_kwargs")
    handler.validate_model_parameters()


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


def write_sse(handler, event_payload):
    handler.wfile.write(
        f"event: {event_payload['type']}\ndata: {json.dumps(event_payload)}\n\n".encode()
    )
    handler.wfile.flush()


def previous_response(handler, previous_response_id, ws_cache=None):
    if not previous_response_id:
        return None
    if ws_cache is not None and previous_response_id in ws_cache:
        return ws_cache[previous_response_id]
    return handler.response_generator.get_open_response(previous_response_id)


def prepare_request(handler, completion_request_cls, ws_cache=None):
    body = handler.body
    previous_response_id = body.get("previous_response_id")
    previous = previous_response(handler, previous_response_id, ws_cache)
    if previous_response_id and previous is None:
        error = ValueError(
            f"Previous response with id '{previous_response_id}' not found."
        )
        error.code = "previous_response_not_found"
        error.param = "previous_response_id"
        raise error

    stored_items = previous.get("items", {}) if previous else {}
    raw_input = body.get("input", "")
    new_items = raw_input if isinstance(raw_input, list) else []
    if previous is not None:
        previous_call_ids = function_call_ids(previous["output"])
        for item in new_items:
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and item.get("call_id") not in previous_call_ids
            ):
                error = ValueError(
                    f"Function call output references unknown call_id '{item.get('call_id')}'."
                )
                error.code = "invalid_request"
                error.param = "input"
                raise error

    messages, input_items = input_to_messages(
        raw_input,
        instructions=body.get("instructions"),
        stored_items=stored_items,
    )
    if previous is not None:
        combined_items = previous["input"] + previous["output"] + input_items
        messages, _ = input_to_messages(
            combined_items,
            stored_items={
                **stored_items,
                **{
                    item.get("id"): item
                    for item in combined_items
                    if isinstance(item, dict) and item.get("id")
                },
            },
        )
        input_items = combined_items

    validate_tool_choice(body.get("tools"), body.get("tool_choice"))
    tools = chat_tools(body.get("tools"), body.get("tool_choice"))
    return (
        completion_request_cls("chat", "", messages, tools, body.get("role_mapping")),
        input_items,
        previous_response_id,
    )


def generation_args(
    handler,
    generation_arguments_cls,
    model_description_cls,
    sampling_arguments_cls,
    logits_processor_arguments_cls,
):
    return generation_arguments_cls(
        model=model_description_cls(
            model=handler.requested_model,
            draft=handler.requested_draft_model,
            adapter=handler.adapter,
        ),
        sampling=sampling_arguments_cls(
            temperature=handler.temperature,
            top_p=handler.top_p,
            top_k=handler.top_k,
            min_p=handler.min_p,
            xtc_probability=handler.xtc_probability,
            xtc_threshold=handler.xtc_threshold,
        ),
        logits=logits_processor_arguments_cls(
            logit_bias=handler.logit_bias,
            repetition_penalty=handler.repetition_penalty,
            repetition_context_size=handler.repetition_context_size,
            presence_penalty=handler.presence_penalty,
            presence_context_size=handler.presence_context_size,
            frequency_penalty=handler.frequency_penalty,
            frequency_context_size=handler.frequency_context_size,
        ),
        stop_words=[],
        max_tokens=handler.max_tokens,
        num_draft_tokens=handler.num_draft_tokens,
        logprobs=handler.logprobs,
        top_logprobs=handler.top_logprobs,
        seed=handler.seed,
        chat_template_kwargs=handler.chat_template_kwargs,
    )


def store_response(handler, response, input_items, ws_cache=None):
    stored = stored_response(response, input_items)
    if ws_cache is not None:
        ws_cache[response["id"]] = stored
        while len(ws_cache) > 16:
            ws_cache.pop(next(iter(ws_cache)))
    if handler.body.get("store", True):
        handler.response_generator.store_open_response(response, input_items)


def run_response(handler, deps, send_event=None, ws_cache=None):
    (
        completion_request_cls,
        generation_arguments_cls,
        model_description_cls,
        sampling_arguments_cls,
        logits_processor_arguments_cls,
        tool_call_formatter_cls,
    ) = deps
    response_id = f"resp_{uuid.uuid4().hex}"
    handler.request_id = response_id
    request, input_items, previous_response_id = prepare_request(
        handler, completion_request_cls, ws_cache
    )
    args = generation_args(
        handler,
        generation_arguments_cls,
        model_description_cls,
        sampling_arguments_cls,
        logits_processor_arguments_cls,
    )
    sequence_number = 0

    def emit(event_type, **kwargs):
        nonlocal sequence_number
        sequence_number += 1
        event_payload = event(event_type, sequence_number, **kwargs)
        if send_event is not None:
            send_event(event_payload)
        return event_payload

    initial_response = response_object(
        handler,
        response_id,
        [],
        usage(0, 0),
        status="queued",
        previous_response_id=previous_response_id,
    )
    emit("response.created", response=initial_response)
    in_progress = dict(initial_response, status="in_progress", completed_at=None)
    emit("response.in_progress", response=in_progress)

    ctx, response = handler.response_generator.generate(request, args)
    tool_formatter = tool_call_formatter_cls(ctx.tool_parser, request.tools, False)

    prev_state = None
    finish_reason = "stop"
    reasoning_text = ""
    tool_text = ""
    tool_calls = []
    output = []
    text = ""
    tokens = []
    message_id = f"msg_{uuid.uuid4().hex}"
    message_started = False

    def start_message():
        nonlocal message_started
        if message_started:
            return
        message_started = True
        emit(
            "response.output_item.added",
            output_index=len(output),
            item=output_message("", message_id, status="in_progress"),
        )
        emit(
            "response.content_part.added",
            item_id=message_id,
            output_index=len(output),
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        )

    def add_tool_calls(raw_tool_calls):
        for tool_call in tool_formatter(raw_tool_calls):
            max_tool_calls = handler.body.get("max_tool_calls")
            if max_tool_calls is not None and len(tool_calls) > max_tool_calls:
                break
            item = tool_call_item(tool_call)
            output.append(item)
            emit(
                "response.output_item.added",
                output_index=len(output) - 1,
                item=dict(item, status="in_progress"),
            )
            emit(
                "response.output_item.done",
                output_index=len(output) - 1,
                item=item,
            )

    try:
        for gen in response:
            if gen.state == "reasoning":
                reasoning_text += gen.text
            elif gen.state == "tool":
                tool_text += gen.text
            elif gen.state == "normal":
                if prev_state == "tool":
                    tool_calls.append(tool_text)
                    add_tool_calls([tool_text])
                    tool_text = ""
                if gen.text:
                    start_message()
                    text += gen.text
                    emit(
                        "response.output_text.delta",
                        item_id=message_id,
                        output_index=len(output),
                        content_index=0,
                        delta=gen.text,
                    )
            tokens.append(gen.token)
            if gen.finish_reason is not None:
                finish_reason = gen.finish_reason
            prev_state = gen.state

        if prev_state == "tool" and tool_text:
            tool_calls.append(tool_text)
            add_tool_calls([tool_text])

        if message_started or not output:
            message = output_message(text, message_id)
            output.append(message)
            emit(
                "response.output_text.done",
                item_id=message_id,
                output_index=len(output) - 1,
                content_index=0,
                text=text,
            )
            emit(
                "response.content_part.done",
                item_id=message_id,
                output_index=len(output) - 1,
                content_index=0,
                part=message["content"][0],
            )
            emit(
                "response.output_item.done",
                output_index=len(output) - 1,
                item=message,
            )
        if reasoning_text:
            output.append(
                {
                    "id": f"rs_{uuid.uuid4().hex}",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": reasoning_text}],
                    "content": [],
                    "encrypted_content": None,
                }
            )

        status = "incomplete" if finish_reason == "length" else "completed"
        error = None
        if tool_call_required(handler.body.get("tool_choice")) and not any(
            item.get("type") == "function_call" for item in output
        ):
            status = "failed"
            error = {
                "code": "tool_required",
                "message": (
                    "tool_choice requires a function call, "
                    "but the model did not call a tool."
                ),
            }
        response_usage = usage(len(ctx.prompt), len(tokens), ctx.prompt_cache_count, 0)
        final_response = response_object(
            handler,
            response_id,
            output,
            response_usage,
            status=status,
            error=error,
            previous_response_id=previous_response_id,
        )
        if status == "incomplete":
            final_response["incomplete_details"] = {"reason": "max_output_tokens"}
        store_response(handler, final_response, input_items, ws_cache)
        emit(f"response.{status}", response=final_response)
        return final_response
    finally:
        ctx.stop()


def handle_response(handler, deps):
    if handler.stream:
        handler._set_stream_headers(200)
        handler.end_headers()
        try:
            run_response(handler, deps, send_event=lambda e: write_sse(handler, e))
            handler.wfile.write("data: [DONE]\n\n".encode())
            handler.wfile.flush()
        except Exception as e:
            code = getattr(e, "code", "invalid_request")
            error_event = event(
                "error",
                1,
                error={
                    "type": "invalid_request",
                    "code": code,
                    "message": str(e),
                    "param": getattr(e, "param", None),
                },
            )
            write_sse(handler, error_event)
            response = response_object(
                handler,
                f"resp_{uuid.uuid4().hex}",
                [],
                usage(0, 0),
                status="failed",
                error={"code": code, "message": str(e)},
                previous_response_id=handler.body.get("previous_response_id"),
            )
            write_sse(handler, event("response.failed", 2, response=response))
            handler.wfile.write("data: [DONE]\n\n".encode())
            handler.wfile.flush()
        return

    try:
        response = run_response(handler, deps)
        response_json = json.dumps(response).encode()
        handler._set_completion_headers(200)
        handler.send_header("Content-Length", str(len(response_json)))
        handler.end_headers()
        handler.wfile.write(response_json)
        handler.wfile.flush()
    except Exception as e:
        code = getattr(e, "code", "invalid_request")
        handler._set_completion_headers(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps(
                error_response(str(e), code=code, param=getattr(e, "param", None))
            ).encode()
        )


def handle_compact(handler):
    if not handler.body.get("model"):
        handler._set_completion_headers(400)
        handler.end_headers()
        handler.wfile.write(
            json.dumps(
                error_response(
                    "Missing required field: model",
                    code="missing_required_parameter",
                    param="model",
                )
            ).encode()
        )
        return
    response = compact_response(handler.body, handler.created)
    response_json = json.dumps(response).encode()
    handler._set_completion_headers(200)
    handler.send_header("Content-Length", str(len(response_json)))
    handler.end_headers()
    handler.wfile.write(response_json)
    handler.wfile.flush()


def websocket_accept(handler):
    key = handler.headers.get("Sec-WebSocket-Key")
    if not key:
        return False
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    ).decode()
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler._set_cors_headers()
    handler.end_headers()
    return True


def websocket_read_frame(handler):
    header = handler.rfile.read(2)
    if len(header) < 2:
        return None, None
    first, second = header
    opcode = first & 0x0F
    masked = second & 0x80
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", handler.rfile.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", handler.rfile.read(8))[0]
    mask = handler.rfile.read(4) if masked else b""
    payload = handler.rfile.read(length)
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def websocket_write_frame(handler, payload, opcode=1):
    if isinstance(payload, str):
        payload = payload.encode()
    length = len(payload)
    header = bytes([0x80 | opcode])
    if length < 126:
        header += bytes([length])
    elif length < (1 << 16):
        header += bytes([126]) + struct.pack("!H", length)
    else:
        header += bytes([127]) + struct.pack("!Q", length)
    handler.wfile.write(header + payload)
    handler.wfile.flush()


def websocket_write_json(handler, payload):
    websocket_write_frame(handler, json.dumps(payload))


def handle_websocket(handler, deps):
    if not websocket_accept(handler):
        handler._set_completion_headers(400)
        handler.end_headers()
        return
    ws_cache = {}
    start = time.time()
    handler.connection.settimeout(1.0)
    while time.time() - start < 3600:
        try:
            opcode, payload = websocket_read_frame(handler)
        except socket.timeout:
            continue
        except OSError:
            break
        if opcode is None or opcode == 8:
            break
        if opcode == 9:
            websocket_write_frame(handler, payload, opcode=10)
            continue
        if opcode != 1:
            continue
        try:
            body = json.loads(payload.decode())
            if body.get("type") != "response.create":
                raise ValueError("WebSocket message type must be response.create")
            for field in ("stream", "stream_options", "background"):
                if field in body:
                    error = ValueError(
                        f"{field} must not be sent on WebSocket response.create"
                    )
                    error.param = field
                    error.code = "invalid_request"
                    raise error
            handler.body = dict(body)
            handler.body.pop("type", None)
            handler.body["stream"] = False
            load_generation_parameters(handler, handler.body)
            run_response(
                handler,
                deps,
                send_event=lambda event_payload: websocket_write_json(
                    handler, event_payload
                ),
                ws_cache=ws_cache,
            )
        except Exception as e:
            previous_response_id = None
            try:
                previous_response_id = body.get("previous_response_id")
            except Exception:
                pass
            if previous_response_id:
                ws_cache.pop(previous_response_id, None)
            websocket_write_json(
                handler,
                {
                    "type": "error",
                    "status": 400,
                    "error": {
                        "type": "invalid_request",
                        "code": getattr(e, "code", "invalid_request"),
                        "message": str(e),
                        "param": getattr(e, "param", None),
                    },
                },
            )
    if time.time() - start >= 3600:
        websocket_write_json(
            handler,
            {
                "type": "error",
                "status": 400,
                "error": {
                    "code": "websocket_connection_limit_reached",
                    "message": "WebSocket connection limit reached.",
                    "param": None,
                },
            },
        )
