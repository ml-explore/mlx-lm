"""Configuration for Residual Precision Streaming."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RPSConfig:
    """Residual Precision Streaming configuration.

    Controls how model weights are decomposed into base + residual tiers
    for memory-efficient inference on constrained hardware.
    """

    base_bits: int = 2
    base_group_size: int = 64
    residual_bits: int = 2
    residual_group_size: int = 64

    # Weight suffixes whose residuals are kept in RAM (Tier 2)
    tier2_keys: List[str] = field(
        default_factory=lambda: ["v_proj", "down_proj"]
    )

    # Tier 3: stream remaining residuals from SSD
    tier3_enabled: bool = False
    tier3_prefetch_depth: int = 2

    # Path to residual safetensors on disk
    residual_path: Optional[str] = None
