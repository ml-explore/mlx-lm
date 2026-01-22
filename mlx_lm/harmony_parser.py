from __future__ import annotations

import re
import uuid
from typing import Any

try:
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    HARMONY_AVAILABLE = True
    _HARMONY_ENCODING = None

    def _get_encoding():
        global _HARMONY_ENCODING
        if _HARMONY_ENCODING is None:
            _HARMONY_ENCODING = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS.value
            )
        return _HARMONY_ENCODING

except ImportError:  # pragma: no cover - optional dependency
    HARMONY_AVAILABLE = False

    def _get_encoding():
        return None


HARMONY_KEYWORDS = ["gpt-oss", "harmony"]


def is_harmony_model(model_name: str) -> bool:
    model_lower = model_name.lower()
    return any(keyword in model_lower for keyword in HARMONY_KEYWORDS)


def _cleanup_harmony_content(content: str) -> str:
    content = re.sub(r"<\|call\|>\s*<\|call\|>", "<|call|>", content)
    content = re.sub(r"<\|end\|>\s*<\|end\|>", "<|end|>", content)
    content = re.sub(
        r"<\|call\|>\s*assistant<\|channel\|>",
        "<|call|><|start|>assistant<|channel|>",
        content,
    )
    return content


_TOOL_CALL_RE = re.compile(
    r"<\|channel\|>commentary\s+to=functions\.(\w+)\s*"
    r"(?:<\|constrain\|>json)?\s*"
    r"<\|message\|>(.*?)<\|call\|>",
    re.DOTALL,
)
_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|$)",
    re.DOTALL,
)
_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|$)",
    re.DOTALL,
)


def _parse_harmony_regex_fallback(content: str) -> dict[str, Any]:
    result = {"tool_calls": [], "reasoning": None, "final_text": ""}

    for match in _TOOL_CALL_RE.finditer(content):
        func_name = match.group(1)
        args_str = match.group(2).strip()
        result["tool_calls"].append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": func_name,
                "arguments": args_str,
            }
        )

    analysis_match = _ANALYSIS_RE.search(content)
    if analysis_match:
        result["reasoning"] = analysis_match.group(1).strip()

    final_match = _FINAL_RE.search(content)
    if final_match:
        result["final_text"] = final_match.group(1).strip()

    return result


def parse_harmony_output(content: str) -> dict[str, Any]:
    result = {"tool_calls": [], "reasoning": None, "final_text": ""}

    if not HARMONY_AVAILABLE:
        result["final_text"] = content
        return result

    encoding = _get_encoding()
    if encoding is None:
        result["final_text"] = content
        return result

    content = _cleanup_harmony_content(content)
    if not content.strip().startswith("<|start|>"):
        content = "<|start|>assistant" + content

    try:
        tokens = encoding.encode(content, allowed_special="all")
        messages = encoding.parse_messages_from_completion_tokens(tokens)

        reasoning_parts = []
        final_parts = []

        for msg in messages:
            channel = getattr(msg, "channel", None)
            recipient = getattr(msg, "recipient", None)
            msg_content = getattr(msg, "content", [])

            text = ""
            for chunk in msg_content:
                if hasattr(chunk, "text"):
                    text = chunk.text
                    break

            if (
                channel == "commentary"
                and recipient
                and recipient.startswith("functions.")
            ):
                func_name = recipient.replace("functions.", "")
                result["tool_calls"].append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "name": func_name,
                        "arguments": text,
                    }
                )
            elif channel == "analysis":
                reasoning_parts.append(text)
            elif channel == "final" or not channel:
                final_parts.append(text)

        result["reasoning"] = "\n".join(reasoning_parts) if reasoning_parts else None
        result["final_text"] = "\n".join(final_parts) if final_parts else ""

    except Exception:
        result = _parse_harmony_regex_fallback(content)

    return result


_HARMONY_TOKEN_RE = re.compile(
    r"<\|channel\|>(?:\w+)?|<\|(?:message|call|end|start|constrain|return|assistant)\|>"
)
_CHANNEL_START_RE = re.compile(r"<\|channel\|>(\w+)")


class HarmonyStreamingParser:
    def __init__(self):
        self.current_channel: str | None = None
        self.buffer: str = ""
        self.full_text: str = ""
        self.reasoning_started: bool = False
        self.message_started: bool = False
        self._awaiting_channel_name: bool = False

    def process_delta(self, delta: str) -> tuple[str | None, str]:
        self.full_text += delta
        text = self.buffer + delta
        self.buffer = ""

        last_open = text.rfind("<|")
        if last_open >= 0:
            close_after = text.find("|>", last_open + 2)
            if close_after < 0:
                self.buffer = text[last_open:]
                text = text[:last_open]

        if self._awaiting_channel_name and text:
            first_word_match = re.match(r"(\w+)", text)
            if first_word_match:
                self.current_channel = first_word_match.group(1).lower()
                text = text[first_word_match.end() :]
            self._awaiting_channel_name = False

        channel_match = _CHANNEL_START_RE.search(text)
        if channel_match:
            self.current_channel = channel_match.group(1).lower()
            self._awaiting_channel_name = False
        elif "<|channel|>" in text:
            self._awaiting_channel_name = True

        clean_text = _HARMONY_TOKEN_RE.sub("", text)

        event_type: str | None = None
        if clean_text:
            if self.current_channel == "analysis":
                event_type = "reasoning"
                self.reasoning_started = True
            elif self.current_channel in ("final", None):
                event_type = "output"
                self.message_started = True

        return event_type, clean_text


__all__ = [
    "HARMONY_AVAILABLE",
    "HarmonyStreamingParser",
    "is_harmony_model",
    "parse_harmony_output",
]
