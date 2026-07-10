# Copyright © 2026 Apple Inc.

"""Model-family-neutral state-budget admission primitives.

The continuous text generator uses these primitives for attention KV caches and
hybrid recurrent/attention state.  Other iterative generators, including
diffusion schedulers, can use the same policy by describing work in denoising
steps and supplying a peak-state projector for their latent and activation
state.  The policy deliberately does not assume that progress is measured in
tokens or that state grows linearly.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass(frozen=True)
class AdmissionState:
    """Scheduler-facing state for one request.

    ``projected_units`` is the total work at completion, while
    ``completed_units`` records current progress.  Units are defined by the
    scheduler (tokens for autoregressive generation, denoising steps for
    diffusion).  ``resident_bytes`` must include state already counted in the
    scheduler's live-byte total.
    """

    uid: Any
    projected_units: int
    completed_units: int = 0
    resident_bytes: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.projected_units < 0 or self.completed_units < 0:
            raise ValueError("work units must be non-negative")
        if self.completed_units > self.projected_units:
            raise ValueError("completed_units cannot exceed projected_units")
        if not math.isfinite(self.resident_bytes) or self.resident_bytes < 0:
            raise ValueError("resident_bytes must be finite and non-negative")


class LinearStateCost:
    """Fixed plus per-unit state cost for dense and hybrid AR caches."""

    def __init__(
        self,
        fixed_bytes: float,
        bytes_per_unit: float,
        max_units: Optional[int] = None,
    ):
        values = (fixed_bytes, bytes_per_unit)
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("state costs must be finite and non-negative")
        if max_units is not None and max_units <= 0:
            raise ValueError("max_units must be positive")
        self.fixed_bytes = fixed_bytes
        self.bytes_per_unit = bytes_per_unit
        self.max_units = max_units

    def __call__(self, state: AdmissionState) -> float:
        units = state.projected_units
        if self.max_units is not None:
            units = min(units, self.max_units)
        return self.fixed_bytes + self.bytes_per_unit * units


class StepStateCost:
    """Peak-state adapter for diffusion and other iterative schedulers.

    The request metadata contains a sequence of total peak byte estimates, one
    per scheduling unit.  For masked diffusion this is commonly one entry per
    denoising step or block and includes the latent/token canvas, logits and
    any prefix/suffix caches.  Admission uses the largest remaining entry, so
    advancing or removing a request releases headroom without assuming linear
    growth or a known number of tokens revealed by each forward.
    """

    def __init__(self, metadata_key: str = "state_bytes_by_step"):
        self.metadata_key = metadata_key

    def __call__(self, state: AdmissionState) -> float:
        try:
            schedule = state.metadata[self.metadata_key]
        except KeyError as error:
            raise ValueError(
                f"AdmissionState.metadata requires {self.metadata_key!r}"
            ) from error
        if len(schedule) < state.projected_units:
            raise ValueError(
                f"{self.metadata_key!r} has {len(schedule)} entries, fewer than "
                f"projected_units={state.projected_units}"
            )
        remaining = schedule[state.completed_units : state.projected_units]
        if any(not math.isfinite(value) or value < 0 for value in remaining):
            raise ValueError(
                f"{self.metadata_key!r} entries must be finite and non-negative"
            )
        return max(remaining, default=state.resident_bytes)


class StateBudget:
    """Admission policy over arbitrary model state.

    ``project`` returns the peak bytes a request can occupy over its remaining
    lifetime, not merely its bytes at the next step.  This distinction lets a
    diffusion adapter account for timestep-dependent activation peaks without
    pretending that its state is a token-linear KV cache.
    """

    def __init__(self, budget_bytes: float, project: Callable[[AdmissionState], float]):
        if not math.isfinite(budget_bytes) or budget_bytes <= 0:
            raise ValueError("budget_bytes must be finite and positive")
        if not callable(project):
            raise TypeError("project must be callable")
        self.budget_bytes = budget_bytes
        self.project = project

    def projected_bytes(self, state: AdmissionState) -> float:
        projected = self.project(state)
        if not math.isfinite(projected) or projected < 0:
            raise ValueError("projected state bytes must be finite and non-negative")
        return projected

    def remaining_bytes(self, state: AdmissionState) -> float:
        return max(self.projected_bytes(state) - state.resident_bytes, 0.0)

    def admitted_prefix(
        self,
        candidates: Iterable[AdmissionState],
        *,
        live_bytes: float,
        active: Iterable[AdmissionState] = (),
        allow_oversized_if_idle: bool = True,
    ) -> int:
        """Return how many candidates fit, preserving the supplied order.

        A caller that reorders for padding efficiency must pass the exact
        selected order here; a count computed for one cohort cannot constrain a
        different cohort.  Removed requests naturally free headroom because
        all accounting is recomputed from the supplied live state.
        """

        if not math.isfinite(live_bytes) or live_bytes < 0:
            raise ValueError("live_bytes must be finite and non-negative")
        active = tuple(active)
        committed = live_bytes + sum(self.remaining_bytes(x) for x in active)
        admitted = 0
        for state in candidates:
            need = self.remaining_bytes(state)
            if committed + need > self.budget_bytes:
                if allow_oversized_if_idle and admitted == 0 and not active:
                    return 1
                break
            committed += need
            admitted += 1
        return admitted
