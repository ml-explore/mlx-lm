# Copyright © 2026 Apple Inc.

import importlib
from collections.abc import Sequence
from typing import Any, Protocol

from mlx_lm.utils import model_package

__all__ = [
    "ENCODER_CONFIGS",
    "Processor",
    "load_processor",
]

# A checkpoint converted without its encoders has none of these, and takes
# only text.
ENCODER_CONFIGS = ("vision_config", "audio_config")


class Processor(Protocol):
    def __call__(self, text: str, **media: Sequence[Any]) -> tuple[str, dict[str, Any]]:
        """The text with the placeholders expanded, and the encoder inputs."""


def load_processor(config: dict[str, Any]) -> Processor | None:
    """The processor of the model of a configuration, or None if it takes only text."""
    if not any(config.get(key) for key in ENCODER_CONFIGS):
        return None
    try:
        package = model_package(config)
    except ValueError:
        return None
    name = f"{package}.processor"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        # The model has no processor, or is a single module.
        if e.name != name:
            raise
        return None
    return module.Processor.from_config(config)
