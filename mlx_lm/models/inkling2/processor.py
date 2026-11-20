# Copyright © 2026 Apple Inc.

"""Image processor of Inkling 2."""

import math
from dataclasses import dataclass
from typing import BinaryIO, ClassVar, Sequence

import mlx.core as mx
import numpy as np

__all__ = [
    "IMAGE_TOKEN",
    "Processor",
]

IMAGE_TOKEN = "<|unused_200054|>"


def _composite_alpha(rgba: np.ndarray) -> np.ndarray:
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153)[..., None]
    rgb, alpha = rgba[..., :3].astype(np.int32), rgba[..., 3:].astype(np.int32)
    blended = (rgb * alpha + background * (255 - alpha) + 127) // 255
    return blended.astype(np.uint8)


@dataclass(frozen=True)
class Processor:
    image_token: ClassVar[str] = IMAGE_TOKEN

    block_size: int = 40
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_config(cls, config: dict) -> "Processor":
        vision = config.get("vision_config")
        if not vision:
            raise ValueError("The model config has no vision_config.")
        patch_size = vision.get("native_patch_size", 20)
        merge_size = vision.get("adapter_merge_size", 2)
        return cls(block_size=patch_size * merge_size)

    def load_image(self, fp: str | BinaryIO) -> np.ndarray:
        """A uint8 (H, W, 3) image. As the reference, transparent pixels are
        put on a gray checkerboard."""
        from PIL import Image

        with Image.open(fp) as image:
            if not image.has_transparency_data:
                return np.asarray(image.convert("RGB"))
            rgba = np.asarray(image.convert("RGBA"))
        return _composite_alpha(rgba)

    def preprocess_image(self, image: np.ndarray) -> tuple[mx.array, int]:
        assert image.ndim == 3 and image.shape[-1] == 3, "Images must be (H, W, 3)."
        assert image.dtype == np.uint8, "Images must be uint8."
        height, width, _ = image.shape
        rows = math.ceil(height / self.block_size)
        cols = math.ceil(width / self.block_size)
        # As the reference, the padding is -1 on the 0-255 scale.
        x = np.full(
            (rows * self.block_size, cols * self.block_size, 3), -1.0, np.float32
        )
        x[:height, :width] = image
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        x = (x * np.float32(1 / 255) - mean) / std
        return mx.array(x.transpose(2, 0, 1)), rows * cols

    def __call__(
        self, text: str, *, images: Sequence[np.ndarray]
    ) -> tuple[str, dict[str, list[mx.array]]]:
        parts = text.split(IMAGE_TOKEN)
        if len(parts) - 1 != len(images):
            raise ValueError(
                f"The text has {len(parts) - 1} {IMAGE_TOKEN} "
                f"for {len(images)} images."
            )
        pixel_values = []
        expanded = [parts[0]]
        for image, part in zip(images, parts[1:], strict=True):
            pixels, num_tokens = self.preprocess_image(image)
            expanded += [IMAGE_TOKEN * num_tokens, part]
            pixel_values.append(pixels)
        return "".join(expanded), {"pixel_values": pixel_values}
