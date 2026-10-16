# Copyright © 2026 Apple Inc.

"""Image processor of Muse Spark."""

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar, Sequence

import mlx.core as mx
import numpy as np

IMAGE_START = "<|image_start|>"
IMAGE_PATCH = "<|patch|>"
IMAGE_END = "<|image_end|>"


def image_tokens(num_tokens: int) -> str:
    return IMAGE_START + IMAGE_PATCH * num_tokens + IMAGE_END


def _least_distortion_fit(
    i_nph: float,
    i_npw: float,
    *,
    img_h: int,
    img_w: int,
    patch_h: int,
    patch_w: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[int, int] | None:
    """The floor/ceil grid with the least aspect distortion, or None."""
    rows = [math.floor(i_nph), math.ceil(i_nph)]
    cols = [math.floor(i_npw), math.ceil(i_npw)]
    candidates = list(set(itertools.product(rows, cols)))
    candidates = [
        (h, w)
        for h, w in candidates
        if h > 0 and w > 0 and min_tokens <= h * w <= max_tokens
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: abs((c[0] * patch_h) / (c[1] * patch_w) - (img_h / img_w)),
    )


def _least_distance_fit(
    i_nph: float,
    i_npw: float,
    *,
    patch_h: int,
    patch_w: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[int, int]:
    """The grid nearest to the ideal grid (L2), within the token bounds."""
    nph = np.arange(1, max_tokens + 1)
    npw_upper = np.floor(max_tokens / nph)
    npw_lower = np.ceil(min_tokens / nph)
    valid = npw_upper >= npw_lower
    if not valid.any():
        raise ValueError(
            f"No grid of image tokens fits min_num_image_tokens={min_tokens} "
            f"and max_num_image_tokens={max_tokens}."
        )
    npw = np.clip(i_npw, npw_lower, npw_upper)[valid]
    nph = nph[valid]
    l2 = ((nph - i_nph) * patch_h) ** 2 + ((npw - i_npw) * patch_w) ** 2
    i = np.argmin(l2)
    return int(nph[i]), int(npw[i])


def _find_resize_target(
    img_h: int,
    img_w: int,
    *,
    patch_h: int,
    patch_w: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The resize target (height, width) and its token grid (rows, columns)."""
    i_nph = img_h / patch_h
    i_npw = img_w / patch_w
    ratio = (img_w / patch_w) / (img_h / patch_h)
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    elif i_nph * i_npw < min_tokens:
        i_nph = (min_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    bounds = dict(
        patch_h=patch_h, patch_w=patch_w, min_tokens=min_tokens, max_tokens=max_tokens
    )
    grid = _least_distortion_fit(i_nph, i_npw, img_h=img_h, img_w=img_w, **bounds)
    nph, npw = grid or _least_distance_fit(i_nph, i_npw, **bounds)
    return (nph * patch_h, npw * patch_w), (nph, npw)


@dataclass(frozen=True)
class Processor:
    # The token of the image patches in the prompt.
    image_token: ClassVar[str] = IMAGE_PATCH

    patch_size: int = 14
    downsample_factor: int = 2
    min_num_image_tokens: int = 1
    max_num_image_tokens: int = 4096
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_config(cls, config: dict) -> "Processor":
        vision = config.get("vision_config")
        if not vision:
            raise ValueError("The model config has no vision_config.")
        # The other settings are in processor_config.json; the defaults match it.
        return cls(
            patch_size=vision.get("patch_size", cls.patch_size),
            downsample_factor=vision.get("merge_size", cls.downsample_factor),
        )

    @property
    def token_patch(self) -> int:
        """The side of one image token, in pixels."""
        return self.patch_size * self.downsample_factor

    def preprocess_image(self, image: np.ndarray) -> tuple[mx.array, int]:
        """The pixel values (3, H, W) and the number of image tokens."""
        from PIL import Image

        assert image.ndim == 3 and image.shape[-1] == 3, "Images must be (H, W, 3)."
        assert image.dtype == np.uint8, "Images must be uint8."
        height, width, _ = image.shape
        (target_h, target_w), (nph, npw) = _find_resize_target(
            height,
            width,
            patch_h=self.token_patch,
            patch_w=self.token_patch,
            min_tokens=self.min_num_image_tokens,
            max_tokens=self.max_num_image_tokens,
        )
        image = Image.fromarray(image).resize(
            (target_w, target_h), resample=Image.Resampling.LANCZOS
        )
        x = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        x = (np.clip(x, 0.0, 1.0) - mean) / std
        return mx.array(x.transpose(2, 0, 1)), nph * npw

    def __call__(
        self, text: str, *, images: Sequence[np.ndarray]
    ) -> tuple[str, list[mx.array]]:
        """Expand each <|patch|> of the chat template to the tokens of one
        image, and preprocess the images."""
        parts = text.split(IMAGE_PATCH)
        if len(parts) - 1 != len(images):
            raise ValueError(
                f"The text has {len(parts) - 1} {IMAGE_PATCH} "
                f"for {len(images)} images."
            )
        pixel_values = []
        expanded = [parts[0]]
        for image, part in zip(images, parts[1:], strict=True):
            pixels, num_tokens = self.preprocess_image(image)
            expanded += [image_tokens(num_tokens), part]
            pixel_values.append(pixels)
        return "".join(expanded), pixel_values
