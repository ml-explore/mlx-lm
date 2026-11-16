# Copyright © 2026 Apple Inc.

"""Image processors of multimodal models, one module per model."""

from mlx import nn

from . import muse_spark

# The processor of each model module.
PROCESSORS = {
    "muse_spark": muse_spark.Processor,
}


def load_processor(model: nn.Module, config: dict) -> muse_spark.Processor | None:
    """The processor of a model, or None."""
    if not config.get("vision_config"):
        return None
    model_module = type(model).__module__.rsplit(".", 1)[-1]
    name = config.get("processor_type", model_module)
    processor = PROCESSORS.get(name)
    return processor.from_config(config) if processor else None
