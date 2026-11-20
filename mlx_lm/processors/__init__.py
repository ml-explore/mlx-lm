# Copyright © 2026 Apple Inc.

import importlib
from collections.abc import Sequence
from typing import Any, BinaryIO, Protocol

import numpy as np

from mlx_lm.utils import model_package

__all__ = [
    "ENCODER_CONFIGS",
    "PROCESSORS",
    "Processor",
    "load_image",
    "load_processor",
]

PROCESSORS = ("inkling2", "muse_spark")

# A checkpoint converted without its encoders has none of these, and takes
# only text.
ENCODER_CONFIGS = ("vision_config", "audio_config")


class Processor(Protocol):
    def __call__(self, text: str, **media: Sequence[Any]) -> tuple[str, dict[str, Any]]:
        """The text with the placeholders expanded, and the encoder inputs."""


def load_processor(config: dict) -> Processor | None:
    """The processor of the model of a configuration, or None if it takes only text."""
    if not any(config.get(key) for key in ENCODER_CONFIGS):
        return None
    if "processor_type" in config:
        name = config["processor_type"]
    else:
        name = model_package(config)
    if name not in PROCESSORS:
        return None
    # Imported here: the package holds the whole model.
    module = importlib.import_module(f"mlx_lm.models.{name}.processor")
    return module.Processor.from_config(config)


def load_image(processor: Processor, fp: str | BinaryIO) -> np.ndarray:
    """A uint8 (H, W, 3) image for the processor."""
    if hasattr(processor, "load_image"):
        return processor.load_image(fp)
    from PIL import Image

    with Image.open(fp) as image:
        return np.asarray(image.convert("RGB"))
