# ABOUTME: Defines request and sequence state used by continuous batching engine.
# ABOUTME: Provides context objects shared between scheduler, runner, and handler.

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SequenceState:
    """Minimal engine-facing state tracking prompt and decode progress."""

    request_id: str
    prompt_len: int
    max_new_tokens: int
    prompt_pos: int = 0
    generated_tokens: int = 0
    finished: bool = False
    cancel_requested: bool = False
    slot_id: Optional[int] = None

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(self.prompt_len - self.prompt_pos, 0)

    @property
    def remaining_generation_tokens(self) -> int:
        return max(self.max_new_tokens - self.generated_tokens, 0)


@dataclass
class SequenceContext:
    """Handler-facing envelope containing SequenceState and streaming metadata."""

    state: SequenceState
    prompt_tokens: List[int]
    sampler_settings: Dict[str, Any]
    stopping_settings: Dict[str, Any]
    tokenizer: Any
    prompt_cache: Optional[Any] = None
    logits_processors: Optional[List[Any]] = None
    sampler: Optional[Any] = None
    start_time_ns: int = field(default_factory=lambda: time.time_ns())
    events: "queue.Queue[Any]" = field(default_factory=lambda: queue.Queue(maxsize=256))
    lock: threading.Lock = field(default_factory=threading.Lock)

    prefill_started_ns: Optional[int] = None
    prefill_completed_ns: Optional[int] = None
    decode_started_ns: Optional[int] = None
    history_tokens: List[int] = field(default_factory=list)
    stop_sequences: List[List[int]] = field(default_factory=list)
    generator_uid: Optional[int] = None
    prompt_inserted: bool = False

    def enqueue_event(self, event: Any) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.state.cancel_requested = True
            self.state.finished = True
