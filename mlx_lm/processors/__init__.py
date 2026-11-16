# Copyright © 2026 Apple Inc.

"""Image processors of multimodal models, one module per model."""

from mlx import nn

from ..utils import model_package
from . import muse_spark

# The processor of each model package.
PROCESSORS = {
    "muse_spark": muse_spark.Processor,
}


def load_processor(model: nn.Module, config: dict) -> muse_spark.Processor | None:
    """The processor of a model, or None."""
    if not config.get("vision_config"):
        return None
    if "processor_type" in config:
        name = config["processor_type"]
    else:
        name = model_package(config)
    processor = PROCESSORS.get(name)
    return processor.from_config(config) if processor else None
