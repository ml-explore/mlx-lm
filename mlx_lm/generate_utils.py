# Copyright © 2023-2024 Apple Inc.

import time
from dataclasses import dataclass, field

import mlx.core as mx

__all__ = ["BatchCounters", "BatchReading", "BatchStats"]


def _peak_memory_gb() -> float:
    """The process-wide peak memory in GB."""
    return mx.get_peak_memory() / 1e9


@dataclass(kw_only=True, repr=True)
class BatchCounters:
    """
    The quantities a generator counts as it works.

    None of them ever decrease, so any two readings of them bound an interval
    and their difference describes it.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_steps (int): The number of decoding steps taken. One step
            emits a token for every sequence being decoded, so this is at most
            ``generation_tokens``.
        decode_time (float): The time in seconds spent inside the model
            decoding. Excludes prompt processing and any caller-side work.
    """

    prompt_tokens: int = 0
    prompt_time: float = 0.0
    generation_tokens: int = 0
    generation_steps: int = 0
    decode_time: float = 0.0

    @property
    def prompt_tps(self) -> float:
        """The prompt processing tokens-per-second."""
        return (
            0.0 if self.prompt_time <= 0.0 else (self.prompt_tokens / self.prompt_time)
        )

    @property
    def decode_tps(self) -> float:
        """Tokens per second spent inside the model.

        Unlike ``generation_tps`` this excludes caller-side work such as
        detokenization, so it measures the speed of decoding alone.
        """
        return (
            0.0
            if self.decode_time <= 0.0
            else (self.generation_tokens / self.decode_time)
        )


@dataclass(kw_only=True, repr=True)
class BatchReading(BatchCounters):
    """
    A generator's counters at a moment in time.

    Constructing one stamps it: ``measured_at`` is set to now and
    ``peak_memory`` is sampled, so a reading is always a complete observation
    rather than counters awaiting a timestamp.

    Args:
        measured_at (float): When the reading was taken. Defaults to now; pass
            a value only to override it, e.g. in tests.
        peak_memory (float): The peak memory used in GB as of this reading.
            Process-wide, so it includes memory the generator did not allocate
            itself.
    """

    measured_at: float = field(default_factory=time.perf_counter)
    peak_memory: float = field(default_factory=_peak_memory_gb)

    @staticmethod
    def between(start: "BatchReading", end: "BatchReading") -> "BatchStats":
        """The work counted between two readings."""
        return BatchStats(
            prompt_tokens=end.prompt_tokens - start.prompt_tokens,
            prompt_time=end.prompt_time - start.prompt_time,
            generation_tokens=end.generation_tokens - start.generation_tokens,
            generation_steps=end.generation_steps - start.generation_steps,
            decode_time=end.decode_time - start.decode_time,
            wall_time=end.measured_at - start.measured_at,
            # A high-water mark rather than a total, so take the larger.
            peak_memory=max(start.peak_memory, end.peak_memory),
        )


@dataclass(kw_only=True, repr=True)
class BatchStats(BatchCounters):
    """
    What a generator did over an interval.

    ``prompt_time``, ``decode_time`` and ``overhead_time`` partition
    ``wall_time`` exactly. Use :obj:`BatchReading.between` to make one.

    Args:
        wall_time (float): The duration of the interval.
        peak_memory (float): The peak memory used in GB over the interval.
            Process-wide, so it includes memory the generator did not allocate
            itself.
    """

    wall_time: float = 0.0
    peak_memory: float = 0.0

    @property
    def generation_time(self) -> float:
        """
        The time in seconds spent generating, excluding prompt processing.

        Includes caller-side work, so it is ``decode_time`` plus
        ``overhead_time``.
        """
        return max(0.0, self.wall_time - self.prompt_time)

    @property
    def generation_tps(self) -> float:
        """The tokens-per-second for generation."""
        if self.generation_time <= 0:
            return 0.0
        return self.generation_tokens / self.generation_time

    @property
    def overhead_time(self) -> float:
        """Interval time spent neither decoding nor processing prompts."""
        return max(0.0, self.wall_time - self.prompt_time - self.decode_time)

    def __iadd__(self, other: "BatchStats") -> "BatchStats":
        self.prompt_tokens += other.prompt_tokens
        self.prompt_time += other.prompt_time
        self.generation_tokens += other.generation_tokens
        self.generation_steps += other.generation_steps
        self.decode_time += other.decode_time
        self.wall_time += other.wall_time
        self.peak_memory = max(self.peak_memory, other.peak_memory)
        return self
