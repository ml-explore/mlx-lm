# ABOUTME: Provides helpers for integrating API handler with batching runtime.
# ABOUTME: Delegates streaming requests to continuous batching path when enabled.

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

from ..generate import GenerationResponse



def maybe_handle_continuous_batching(
    handler,
    *,
    prompt_tokens,
    stop_id_sequences,
    sampler_settings: Dict[str, float],
    stopping_settings: Dict[str, Optional[int]],
    logit_bias: Optional[Dict[int, float]],
    repetition_penalty: Optional[float],
    repetition_context_size: Optional[int],
) -> Optional[Tuple[str, Iterable[GenerationResponse]]]:
    runtime = getattr(handler, "batch_runtime", None)
    if runtime is None or not getattr(handler, "stream", False):
        return None

    request_id, generator = runtime.submit_request(
        prompt_tokens,
        max_new_tokens=handler.max_tokens,
        sampler_settings=sampler_settings,
        stopping_settings=stopping_settings,
        logit_bias=logit_bias,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
    )
    handler.request_id = request_id
    return request_id, generator


__all__ = ["maybe_handle_continuous_batching"]
