from __future__ import annotations

import base64
import io
import re
import threading
from typing import Any, Iterable

from . import collect_system_preamble

_CHATML_SPECIAL_TOKENS_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>")

_VLM_CACHE: dict[str, tuple[Any, Any]] = {}
_VLM_LOCK = threading.Lock()


class _FallbackProcessor:
    def __init__(self, tokenizer, image_processor, detokenizer):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.detokenizer = detokenizer
        self.chat_template = getattr(tokenizer, "chat_template", None)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def process(
        self,
        text=None,
        images=None,
        padding=True,
        return_tensors="np",
        add_special_tokens=False,
        **kwargs,
    ):
        tokenized = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors,
        )
        output = dict(tokenized)

        if images is not None and self.image_processor is not None:
            if not isinstance(images, list):
                images = [images]
            processed = self.image_processor(
                images=images, return_tensors=return_tensors
            )
            if isinstance(processed, dict):
                output.update(processed)
        return output


def _build_fallback_processor(model_path, eos_token_ids=None):
    try:
        from mlx_vlm.tokenizer_utils import load_tokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx-vlm tokenizer utilities unavailable") from exc

    from mlx_vlm import utils as vlm_utils

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers AutoTokenizer unavailable") from exc

    detokenizer_cls = load_tokenizer(model_path, return_tokenizer=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    image_processor = vlm_utils.load_image_processor(model_path, trust_remote_code=True)
    if image_processor is None:
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Unable to load image processor for vision model."
            ) from exc

    criteria = vlm_utils.StoppingCriteria(
        eos_token_ids if eos_token_ids is not None else tokenizer.eos_token_ids,
        tokenizer,
    )
    tokenizer.stopping_criteria = criteria

    detokenizer = detokenizer_cls(tokenizer)
    return _FallbackProcessor(tokenizer, image_processor, detokenizer)


def _get_vlm_backend(model_id: str) -> tuple[Any, Any]:
    with _VLM_LOCK:
        cached = _VLM_CACHE.get(model_id)
    if cached:
        return cached

    try:
        from mlx_vlm import load as load_vlm

        model, processor = load_vlm(model_id)
    except Exception as exc:  # pragma: no cover - optional dependency
        message = str(exc).lower()
        if "torchvision" not in message:
            raise RuntimeError("mlx-vlm is required for vision responses") from exc

        from mlx_vlm import utils as vlm_utils

        model_path = vlm_utils.get_model_path(model_id)
        model = vlm_utils.load_model(model_path)
        eos_token_id = getattr(model.config, "eos_token_id", None)
        processor = _build_fallback_processor(model_path, eos_token_id)

    with _VLM_LOCK:
        _VLM_CACHE.setdefault(model_id, (model, processor))
        return _VLM_CACHE[model_id]


def _decode_base64_image(data: str):
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Pillow is required to decode base64 images") from exc

    if data.startswith("data:"):
        if "," not in data:
            raise ValueError("Invalid data URL for image")
        _, data = data.split(",", 1)

    data = data.strip()
    if not data:
        raise ValueError("Empty base64 image data")

    missing = len(data) % 4
    if missing:
        data += "=" * (4 - missing)

    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _content_has_image(content: Any) -> bool:
    if isinstance(content, dict):
        return content.get("type") == "input_image"
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in content
        )
    return False


def has_image_content(normalised_body: dict[str, Any]) -> bool:
    for turn in normalised_body.get("input", []):
        if not isinstance(turn, dict):
            continue
        if _content_has_image(turn.get("content")):
            return True
    return False


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") in (
            "input_text",
            "output_text",
            "text",
            "reasoning_text",
        ):
            text_value = content.get("text")
            return str(text_value) if text_value is not None else ""
        text_value = content.get("text")
        return str(text_value) if text_value is not None else ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append(part)
                continue
            if isinstance(part, dict):
                if (
                    part.get("type")
                    in (
                        "input_text",
                        "output_text",
                        "text",
                        "reasoning_text",
                    )
                    or "text" in part
                ):
                    text_value = part.get("text")
                    if text_value is not None:
                        parts.append(str(text_value))
                continue
            if part is not None:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def build_messages(normalised_body: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    preamble = collect_system_preamble(normalised_body)
    if preamble:
        messages.append({"role": "system", "content": "\n\n".join(preamble)})

    for turn in normalised_body.get("input", []):
        if not isinstance(turn, dict):
            continue

        role = str(turn.get("role", "user")).lower()
        if role not in {"system", "user", "assistant", "tool", "developer"}:
            role = "user"

        content = turn.get("content")
        text = _extract_text_content(content)

        if text or (role == "user" and _content_has_image(content)):
            messages.append({"role": role, "content": text})

    if not messages:
        messages.append({"role": "user", "content": ""})

    return messages


def extract_image_inputs(normalised_body: dict[str, Any]) -> list[Any]:
    images: list[Any] = []
    for turn in normalised_body.get("input", []):
        if not isinstance(turn, dict):
            continue

        content = turn.get("content", [])
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "input_image":
                continue

            if part.get("image_base64"):
                images.append(_decode_base64_image(str(part["image_base64"])))
                continue

            source = part.get("image_url")
            if isinstance(source, str) and source.startswith("data:"):
                images.append(_decode_base64_image(source))
            elif source:
                images.append(source)

    return images


def _generation_kwargs(normalised_body: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    max_tokens = normalised_body.get("max_output_tokens") or normalised_body.get(
        "max_tokens"
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    temperature = normalised_body.get("temperature")
    if temperature is not None:
        kwargs["temperature"] = temperature

    top_p = normalised_body.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p

    return kwargs


def _build_prompt(
    model: Any, processor: Any, normalised_body: dict[str, Any], images: list[Any]
) -> str:
    try:
        from mlx_vlm import apply_chat_template
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx-vlm is required for vision responses") from exc

    messages = build_messages(normalised_body)
    return apply_chat_template(
        processor,
        model.config,
        messages,
        add_generation_prompt=True,
        num_images=len(images),
    )


def generate(
    model_id: str, normalised_body: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    images = extract_image_inputs(normalised_body)
    if not images:
        raise RuntimeError("Vision request missing images")

    try:
        from mlx_vlm.generate import generate as vlm_generate
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx-vlm is required for vision responses") from exc

    model, processor = _get_vlm_backend(model_id)
    prompt = _build_prompt(model, processor, normalised_body, images)
    kwargs = _generation_kwargs(normalised_body)

    result = vlm_generate(
        model,
        processor,
        prompt,
        image=images,
        **kwargs,
    )

    text = _CHATML_SPECIAL_TOKENS_RE.sub("", (result.text or ""))

    usage = {
        "input_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(result, "generation_tokens", 0) or 0),
        "total_tokens": int(getattr(result, "total_tokens", 0) or 0),
    }

    return text, usage


def stream_generate(model_id: str, normalised_body: dict[str, Any]) -> Iterable[str]:
    images = extract_image_inputs(normalised_body)
    if not images:
        raise RuntimeError("Vision request missing images")

    try:
        from mlx_vlm.generate import stream_generate as vlm_stream_generate
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mlx-vlm is required for vision responses") from exc

    model, processor = _get_vlm_backend(model_id)
    prompt = _build_prompt(model, processor, normalised_body, images)
    kwargs = _generation_kwargs(normalised_body)

    for result in vlm_stream_generate(
        model,
        processor,
        prompt,
        image=images,
        **kwargs,
    ):
        delta = _CHATML_SPECIAL_TOKENS_RE.sub("", result.text or "")
        if delta:
            yield delta


__all__ = [
    "build_messages",
    "extract_image_inputs",
    "generate",
    "has_image_content",
    "stream_generate",
]
