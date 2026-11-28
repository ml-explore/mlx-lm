# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlx_lm import utils
from mlx_lm.models import muse_spark
from mlx_lm.models.muse_spark.processor import Processor
from mlx_lm.processors import load_processor


class TestMuseSparkProcessor(unittest.TestCase):
    def test_preprocess_image(self):
        processor = Processor(max_num_image_tokens=16)
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
        processor = Processor()
        images = [
            np.zeros((56, 84, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        ]
        text, inputs = processor("A <|patch|> and B <|patch|>.", images=images)
        expected = (
            "A <|image_start|>" + "<|patch|>" * 6 + "<|image_end|>"
            " and B <|image_start|><|patch|><|image_end|>."
        )
        self.assertEqual(text, expected)
        self.assertEqual(list(inputs), ["pixel_values"])
        shapes = [p.shape for p in inputs["pixel_values"]]
        self.assertEqual(shapes, [(3, 56, 84), (3, 28, 28)])

    def test_call_errors(self):
        processor = Processor()
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("<|patch|> <|patch|>", images=[image])
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("No image", images=[image])

    def test_load_processor(self):
        self.assertIsNone(load_processor({"model_type": "muse_spark"}))
        self.assertIsNone(load_processor({"vision_config": {}}))
        config = {
            "vision_config": {"patch_size": 16},
            "processor_type": "muse_spark",
        }
        processor = load_processor(config)
        self.assertEqual(processor.patch_size, 16)

    def test_load_processor_of_the_model_package(self):
        config = {"model_type": "muse_spark", "vision_config": {"patch_size": 16}}
        self.assertEqual(load_processor(config).patch_size, 16)

    def test_tokenizer_defaults_of_the_model_package(self):
        with tempfile.TemporaryDirectory() as path:
            with open(Path(path) / "config.json", "w") as f:
                json.dump({"model_type": "muse_spark"}, f)
            defaults = utils._tokenizer_defaults(Path(path))
        self.assertEqual(defaults, muse_spark.TOKENIZER_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
