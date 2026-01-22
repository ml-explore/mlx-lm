from __future__ import annotations

import copy
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Optional

_RESPONSE_STORE_MAX = 1000
_RESPONSE_STORE: "OrderedDict[str, StoredResponse]" = OrderedDict()
_RESPONSE_STORE_LOCK = Lock()

_TEXT_PART_TYPES = {"text", "input_text", "output_text", "reasoning_text"}
_IMAGE_PART_TYPES = {"input_image", "image_url", "image_base64"}
_AUDIO_PART_TYPES = {"input_audio", "audio_url"}
_VIDEO_PART_TYPES = {"input_video", "video_url"}
_DEFAULT_MODALITIES = ["text"]


@dataclass
class StoredResponse:
    response: dict
    request: dict
    status: str
    created_at: int
    updated_at: int


def store_response(
    response_id: str, request_payload: dict, response_payload: dict
) -> None:
    now = int(time.time())
    with _RESPONSE_STORE_LOCK:
        _RESPONSE_STORE[response_id] = StoredResponse(
            response=response_payload,
            request=request_payload,
            status=response_payload.get("status", "completed"),
            created_at=response_payload.get("created_at", now),
            updated_at=now,
        )
        _RESPONSE_STORE.move_to_end(response_id)
        while len(_RESPONSE_STORE) > _RESPONSE_STORE_MAX:
            _RESPONSE_STORE.popitem(last=False)


def get_stored_response(response_id: str) -> Optional[StoredResponse]:
    with _RESPONSE_STORE_LOCK:
        return _RESPONSE_STORE.get(response_id)


def delete_stored_response(response_id: str) -> bool:
    with _RESPONSE_STORE_LOCK:
        return _RESPONSE_STORE.pop(response_id, None) is not None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _normalize_part_dict(part: dict) -> Optional[dict]:
    part_type = part.get("type")

    if part_type in _TEXT_PART_TYPES or (part_type is None and "text" in part):
        text_value = part.get("text")
        if text_value is None:
            return None
        return {"type": "input_text", "text": _stringify(text_value)}

    if part_type in _IMAGE_PART_TYPES or "image_url" in part or "image_base64" in part:
        image_base64 = part.get("image_base64")
        image_url = part.get("image_url") or part.get("url") or part.get("file_id")

        if isinstance(image_base64, dict):
            image_base64 = (
                image_base64.get("data")
                or image_base64.get("url")
                or image_base64.get("file_id")
            )
        if isinstance(image_url, dict):
            image_url = image_url.get("url") or image_url.get("file_id")

        if not image_base64 and not image_url:
            return None

        normalized = {"type": "input_image"}
        if image_base64:
            normalized["image_base64"] = image_base64
        elif isinstance(image_url, str) and image_url.startswith("data:"):
            normalized["image_base64"] = image_url
        else:
            normalized["image_url"] = image_url

        if "detail" in part:
            normalized["detail"] = part["detail"]
        return normalized

    if part_type in _AUDIO_PART_TYPES or "audio_url" in part:
        source = part.get("audio_url") or part.get("file_id")
        if not source:
            return None
        return {"type": "input_audio", "audio_url": source}

    if part_type in _VIDEO_PART_TYPES or "video_url" in part:
        source = part.get("video_url") or part.get("file_id")
        if not source:
            return None
        return {"type": "input_video", "video_url": source}

    if part:
        return {"type": "input_text", "text": _stringify(part)}
    return None


def _normalize_content_parts(content: Any) -> List[dict]:
    parts: List[dict] = []

    if content is None:
        return parts

    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    if isinstance(content, dict):
        normalized = _normalize_part_dict(content)
        return [normalized] if normalized else []

    if isinstance(content, (list, tuple)):
        for raw in content:
            if isinstance(raw, str):
                parts.append({"type": "input_text", "text": raw})
            elif isinstance(raw, dict):
                normalized = _normalize_part_dict(raw)
                if normalized:
                    parts.append(normalized)
            elif raw is not None:
                parts.append({"type": "input_text", "text": _stringify(raw)})
        return parts

    return [{"type": "input_text", "text": _stringify(content)}]


def _build_turn(role: str, content: Any) -> dict:
    return {"role": role, "content": _normalize_content_parts(content)}


def _normalize_input(raw_input: Any) -> List[dict]:
    turns: List[dict] = []

    if isinstance(raw_input, str):
        turns.append(_build_turn("user", raw_input))
    elif isinstance(raw_input, dict):
        role = raw_input.get("role", "user")
        turns.append(_build_turn(role, raw_input.get("content")))
    elif isinstance(raw_input, (list, tuple)):
        for entry in raw_input:
            if isinstance(entry, dict):
                role = entry.get("role", "user")
                turns.append(_build_turn(role, entry.get("content")))
            elif isinstance(entry, str):
                turns.append(_build_turn("user", entry))
            elif entry is not None:
                turns.append(_build_turn("user", _stringify(entry)))
    elif raw_input is not None:
        turns.append(_build_turn("user", _stringify(raw_input)))

    if not turns:
        turns.append(_build_turn("user", ""))
    return turns


def _normalize_modalities(modalities: Any, default: List[str]) -> List[str]:
    if isinstance(modalities, list) and modalities:
        return [str(mod).lower() for mod in modalities if mod]
    if isinstance(modalities, str) and modalities.strip():
        return [modalities.strip().lower()]
    return list(default)


def normalize_responses_payload(body: dict) -> dict:
    payload = copy.deepcopy(body)
    payload["input"] = _normalize_input(body.get("input"))
    payload.setdefault("modalities", list(_DEFAULT_MODALITIES))
    payload["modalities"] = _normalize_modalities(
        payload.get("modalities"), _DEFAULT_MODALITIES
    )
    payload["output_modalities"] = _normalize_modalities(
        payload.get("output_modalities"), payload["modalities"]
    )
    return payload


def has_media_content(normalized_body: dict) -> bool:
    modalities = set(normalized_body.get("modalities", []) or [])
    output_modalities = set(normalized_body.get("output_modalities", []) or [])
    media_modalities = {"image", "audio", "video"}

    if media_modalities & (modalities | output_modalities):
        return True

    for turn in normalized_body.get("input", []):
        if not isinstance(turn, dict):
            continue
        contents = turn.get("content", [])
        for part in contents:
            if isinstance(part, dict) and part.get("type") in {
                "input_image",
                "input_audio",
                "input_video",
            }:
                return True
    return False


def parts_to_plaintext(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, (list, tuple)):
        return _stringify(parts)

    lines: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            lines.append(_stringify(part))
            continue
        part_type = part.get("type")
        if part_type == "input_text" and part.get("text"):
            lines.append(_stringify(part["text"]))
        elif part_type == "input_image":
            url = part.get("image_url") or part.get("image_base64")
            if url:
                lines.append(
                    f"[Image: {str(url)[:50]}...]"
                    if len(str(url)) > 50
                    else f"[Image: {url}]"
                )
        elif part_type == "input_audio":
            url = part.get("audio_url")
            if url:
                lines.append(f"[Audio: {url}]")
        elif part_type == "input_video":
            url = part.get("video_url")
            if url:
                lines.append(f"[Video: {url}]")
        elif part_type and part_type.startswith("output_"):
            text = part.get("text")
            if text:
                lines.append(_stringify(text))
        elif "text" in part:
            lines.append(_stringify(part["text"]))

    return "\n".join(lines)


def collect_system_preamble(body: dict) -> List[str]:
    preamble: List[str] = []

    system_instruction = body.get("system_instruction") or body.get("instructions")
    if isinstance(system_instruction, str) and system_instruction.strip():
        preamble.append(system_instruction.strip())
    elif isinstance(system_instruction, dict):
        preamble.append(_stringify(system_instruction))

    formatter = body.get("text", {}).get("format")
    if formatter:
        schema_text = _stringify(formatter)
        preamble.append(
            "When responding, conform strictly to the following JSON schema. "
            "Do not include any prose outside the JSON.\n" + schema_text
        )

    reasoning = body.get("reasoning")
    if reasoning:
        preamble.append("Reasoning guidance: " + _stringify(reasoning))

    return preamble


def responses_to_chat_messages(normalized_body: dict) -> List[dict]:
    messages: List[dict] = []
    preamble = collect_system_preamble(normalized_body)
    if preamble:
        messages.append({"role": "system", "content": "\n\n".join(preamble)})

    for turn in normalized_body.get("input", []):
        if not isinstance(turn, dict):
            continue
        text = parts_to_plaintext(turn.get("content"))
        if not text:
            continue
        role = str(turn.get("role", "user")).lower()
        if role not in {"system", "user", "assistant", "tool", "developer"}:
            role = "user"
        messages.append({"role": role, "content": text})

    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def _filter_media_from_content(content: Any) -> List[dict]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": _stringify(content)}]

    filtered: List[dict] = []
    media_types = {"input_image", "input_audio", "input_video", "image_url"}

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type in media_types:
            continue
        if part_type in {"input_text", "text"}:
            filtered.append(part)
    return filtered


def _extract_text_from_output(output: List[dict]) -> str:
    texts: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type == "message":
            content = item.get("content", [])
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in (
                            "text",
                            "output_text",
                            "reasoning_text",
                        ):
                            texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
        elif item_type in ("text", "output_text", "reasoning_text"):
            texts.append(item.get("text", ""))
    return "\n".join(filter(None, texts))


def build_response_output_items(
    text: str,
    reasoning_text: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> List[dict]:
    items: List[dict] = []

    if reasoning_text:
        items.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            }
        )

    items.append(
        {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
    )

    for tool_call in tool_calls or []:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
        if not call_id:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
        items.append(
            {
                "id": call_id,
                "type": "function_call",
                "name": function.get("name"),
                "arguments": function.get("arguments", ""),
                "call_id": call_id,
                "status": "completed",
            }
        )

    return items


def build_context_from_previous_response(
    stored: StoredResponse, current_input: List[dict]
) -> List[dict]:
    messages: List[dict] = []

    prev_input = stored.request.get("input", [])
    if isinstance(prev_input, list):
        for item in prev_input:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            filtered_content = _filter_media_from_content(item.get("content", []))
            if filtered_content:
                messages.append({"role": role, "content": filtered_content})

    assistant_text = _extract_text_from_output(stored.response.get("output", []))
    if assistant_text:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            }
        )

    messages.extend(current_input)
    return messages


__all__ = [
    "StoredResponse",
    "collect_system_preamble",
    "build_context_from_previous_response",
    "build_response_output_items",
    "delete_stored_response",
    "get_stored_response",
    "has_media_content",
    "normalize_responses_payload",
    "responses_to_chat_messages",
    "store_response",
]
