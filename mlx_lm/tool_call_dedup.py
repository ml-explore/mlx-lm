"""Consecutive duplicate tool call detection.

Tracks raw tool call text as it is appended during generation.
When the same text appears consecutively, signals the caller to stop
generation early — preventing degenerate loops where the model
produces identical tool calls until max_tokens.

See: https://github.com/ml-explore/mlx-lm/issues/613
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_LOG_LEN = 120


class ToolCallDedup:
    """Detect consecutive duplicate tool calls during generation.

    Usage::

        dedup = ToolCallDedup()
        # After each tool_call_end token:
        if dedup.is_duplicate(tool_text):
            # stop generation
            ...
        else:
            tool_calls.append(tool_text)
    """

    def __init__(self) -> None:
        self._prev: str | None = None

    def is_duplicate(self, tool_text: str) -> bool:
        """Return True if *tool_text* matches the previous call exactly."""
        if self._prev is not None and tool_text == self._prev:
            logger.warning(
                "Consecutive duplicate tool call detected, stopping: %s",
                tool_text[:_MAX_LOG_LEN],
            )
            return True
        self._prev = tool_text
        return False
