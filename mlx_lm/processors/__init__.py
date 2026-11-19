# Copyright © 2026 Apple Inc.

import importlib.util
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
    if "processor_type" in config:
        package = f"mlx_lm.models.{config['processor_type']}"
    else:
        package = model_package(config)
    name = f"{package}.processor"
    # Imported here: the package holds the whole model.
    if importlib.util.find_spec(name) is None:
        return None
    module = importlib.import_module(name)
    return module.Processor.from_config(config)
