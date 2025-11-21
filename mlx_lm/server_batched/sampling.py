# ABOUTME: Provides vectorized sampling helpers for batched decode.
# ABOUTME: Supplies utilities to pick tokens and logprobs per batch row.

from __future__ import annotations

from typing import Tuple

import numpy as np


def select_tokens_argmax(logits: np.ndarray) -> np.ndarray:
    """Return argmax token ids per batch row.

    Args:
        logits: np.ndarray shaped [B, V] containing logits.

    Returns:
        np.ndarray int64 shaped [B] with argmax indices per row.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be rank-2 [B, V]")
    return logits.argmax(axis=-1)


def selected_logprobs(logits: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Compute log-softmax for selected tokens per row.

    This avoids constructing the full log-softmax matrix and instead computes
    logprob(token) - logsumexp(logits_row).
    """
    if logits.ndim != 2:
        raise ValueError("logits must be rank-2")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("token_ids must be rank-1 with same batch size as logits")

    # Stable logsumexp per row
    max_logits = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logits
    logsumexp = np.log(np.exp(shifted).sum(axis=-1)) + max_logits.squeeze(-1)

    picked = logits[np.arange(logits.shape[0]), token_ids]
    return picked - logsumexp


__all__ = ["select_tokens_argmax", "selected_logprobs"]
