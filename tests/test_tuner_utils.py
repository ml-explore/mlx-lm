# Copyright © 2024 Apple Inc.

import sys
import unittest
import warnings
from io import StringIO
from math import ceil
from unittest.mock import MagicMock

import mlx.nn as nn

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import build_schedule_config, print_trainable_parameters


class TestTunerUtils(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = StringIO()
        sys.stdout = self.capturedOutput

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_quantized_print_trainable_parameters(self):
        model = MagicMock()
        quantized_linear = MagicMock(spec=nn.QuantizedLinear)
        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 8
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=2e6)
        lora_linear.parameters.return_value = [lora_linear.weight]

        linear = MagicMock(spec=nn.Linear)
        linear.weight = MagicMock(size=3e6)
        linear.parameters.return_value = [linear.weight]

        model.leaf_modules.return_value = {
            "quantized_linear": quantized_linear,
            "lora_linear": lora_linear,
            "linear": linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output_8bits = "Trainable parameters: 33.333% (3.000M/9.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_8bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 4
        expected_output_4bits = "Trainable parameters: 23.077% (3.000M/13.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_4bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

    def test_print_trainable_parameters(self):
        model = MagicMock()
        linear1 = MagicMock(spec=nn.Linear)
        linear1.weight = MagicMock(size=1e6)
        linear1.parameters.return_value = [linear1.weight]
        linear2 = MagicMock(spec=nn.Linear)
        linear2.weight = MagicMock(size=2e6)
        linear2.parameters.return_value = [linear2.weight]
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=3e6)
        lora_linear.parameters.return_value = [lora_linear.weight]
        model.leaf_modules.return_value = {
            "linear1": linear1,
            "linear2": linear2,
            "lora_linear": lora_linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output)

    def test_schedule_config_accepts_alias_and_mlx_names(self):
        cfg_alias = build_schedule_config(
            learning_rate=1e-4,
            iters=100,
            lr_schedule="cosine",
            lr_steps=None,
            warmup_steps=10,
        )
        self.assertEqual(cfg_alias["name"], "cosine_decay")

        cfg_mlx = build_schedule_config(
            learning_rate=1e-4,
            iters=100,
            lr_schedule="cosine_decay",
            lr_steps=None,
            warmup_steps=10,
        )
        self.assertEqual(cfg_mlx["name"], "cosine_decay")

    def test_schedule_config_unknown_warns_and_falls_back_to_constant(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = build_schedule_config(
                learning_rate=1e-4,
                iters=100,
                lr_schedule="nope",
                warmup_steps=10,
            )
            self.assertEqual(cfg["name"], "constant")
            self.assertTrue(
                any(
                    "Unknown value for --lr_schedule" in str(item.message) for item in w
                )
            )

    def test_schedule_config_converts_steps_to_updates_with_accum(self):
        cfg = build_schedule_config(
            learning_rate=1e-4,
            iters=100,
            lr_schedule="linear",
            lr_steps=None,
            lr_end=0.0,
            warmup_steps=20,
            warmup_init=1e-7,
            grad_accumulation_steps=4,
        )
        self.assertEqual(cfg["name"], "linear_schedule")
        self.assertEqual(cfg["warmup"], ceil(20 / 4))
        self.assertEqual(cfg["arguments"][2], ceil(100 / 4) - ceil(20 / 4))

        cfg2 = build_schedule_config(
            learning_rate=1e-4,
            iters=100,
            lr_schedule="linear",
            lr_steps=50,
            lr_end=0.0,
            warmup_steps=10,
            grad_accumulation_steps=5,
        )
        self.assertEqual(cfg2["arguments"][2], ceil(50 / 5) - ceil(10 / 5))

    def test_step_decay_step_size_converted_with_accum_or_defaulted(self):
        cfg = build_schedule_config(
            learning_rate=1e-4,
            iters=100,
            lr_schedule="step",
            lr_decay_rate=0.5,
            lr_step_size=8,
            grad_accumulation_steps=4,
        )
        self.assertEqual(cfg["name"], "step_decay")
        self.assertEqual(cfg["arguments"][2], ceil(8 / 4))

        cfg2 = build_schedule_config(
            learning_rate=1e-4,
            iters=100,
            lr_schedule="step",
            lr_decay_rate=0.5,
            grad_accumulation_steps=5,
        )
        self.assertEqual(cfg2["arguments"][2], max(1, (ceil(100 / 5)) // 10))


if __name__ == "__main__":
    unittest.main()
