# Copyright © 2026 Apple Inc.

from collections.abc import Sequence
from typing import Any, Protocol

from mlx_lm.models import model_package
from mlx_lm.plugins.discovery import PluginKind, import_if_exists

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
    """The processor of a built-in model, or None if it has none or takes only text."""
    if not any(config.get(key) for key in ENCODER_CONFIGS):
        return None
    package = PluginKind.MODELS.builtin_module(model_package(config))
    if package is None:
        return None
    module = import_if_exists(f"{package.__name__}.processor")
    if module is None:
        return None
    return module.Processor.from_config(config)
