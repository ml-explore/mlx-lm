# Copyright © 2026 Apple Inc.

"""Image processors of multimodal models, one module per model."""

from typing import BinaryIO

import numpy as np
from mlx import nn

from . import inkling2, muse_spark

# The processor of each model module.
PROCESSORS = {
    "inkling2": inkling2.Processor,
    "muse_spark": muse_spark.Processor,
}


def load_processor(
    model: nn.Module, config: dict
) -> inkling2.Processor | muse_spark.Processor | None:
    """The processor of a model, or None."""
    if not config.get("vision_config"):
        return None
    model_module = type(model).__module__.rsplit(".", 1)[-1]
    name = config.get("processor_type", model_module)
    processor = PROCESSORS.get(name)
    return processor.from_config(config) if processor else None


def load_image(
    processor: inkling2.Processor | muse_spark.Processor, fp: str | BinaryIO
) -> np.ndarray:
    """A uint8 (H, W, 3) image for the processor."""
    if hasattr(processor, "load_image"):
        return processor.load_image(fp)
    from PIL import Image

    with Image.open(fp) as image:
        return np.asarray(image.convert("RGB"))
