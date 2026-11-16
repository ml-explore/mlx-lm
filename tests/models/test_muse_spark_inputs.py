# Copyright © 2026 Apple Inc.

import unittest

import numpy as np
from mlx import nn

from mlx_lm.processors import load_processor, muse_spark


class TestMuseSparkProcessor(unittest.TestCase):
    def test_preprocess_image(self):
        processor = muse_spark.Processor(max_num_image_tokens=16)
        # (height, width), pixel shape, number of tokens
        test_cases = [
            # A multiple of 28, no resize.
            ((56, 84), (3, 56, 84), 6),
            # Resized to the nearest grid with the same aspect ratio.
            ((100, 50), (3, 112, 56), 8),
            # Resized down to max_num_image_tokens.
            ((280, 280), (3, 112, 112), 16),
        ]
        for size, shape, num_tokens in test_cases:
            with self.subTest(size=size):
                image = np.zeros((*size, 3), dtype=np.uint8)
                pixels, n = processor.preprocess_image(image)
                self.assertEqual(pixels.shape, shape)
                self.assertEqual(n, num_tokens)

    def test_call(self):
        processor = muse_spark.Processor()
        images = [
            np.zeros((56, 84, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        ]
        text, pixel_values = processor("A <|patch|> and B <|patch|>.", images=images)
        expected = (
            "A <|image_start|>" + "<|patch|>" * 6 + "<|image_end|>"
            " and B <|image_start|><|patch|><|image_end|>."
        )
        self.assertEqual(text, expected)
        self.assertEqual([p.shape for p in pixel_values], [(3, 56, 84), (3, 28, 28)])

    def test_call_errors(self):
        processor = muse_spark.Processor()
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("<|patch|> <|patch|>", images=[image])
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("No image", images=[image])

    def test_load_processor(self):
        model = nn.Module()
        self.assertIsNone(load_processor(model, {"model_type": "muse_spark"}))
        self.assertIsNone(load_processor(model, {"vision_config": {}}))
        config = {
            "vision_config": {"patch_size": 16},
            "processor_type": "muse_spark",
        }
        processor = load_processor(model, config)
        self.assertEqual(processor.patch_size, 16)


if __name__ == "__main__":
    unittest.main()
