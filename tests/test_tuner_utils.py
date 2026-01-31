# Copyright © 2024 Apple Inc.

import math
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock

import mlx.nn as nn
import mlx.optimizers as opt

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import build_schedule, print_trainable_parameters


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


class TestScheduleArguments(unittest.TestCase):
    def test_linear_schedule_argument_variants(self):
        # Only start provided vs explicit end and steps
        sched_a = build_schedule(
            {"name": "linear_schedule", "arguments": [0.1]}, iters=10
        )
        sched_b = build_schedule(
            {"name": "linear_schedule", "arguments": [0.1, 0.0, 10]}
        )
        for t in (0, 5, 10):
            self.assertAlmostEqual(sched_a(t), sched_b(t), delta=1e-8)

        # Start and end provided vs explicit steps
        sched_c = build_schedule(
            {"name": "linear_schedule", "arguments": [0.2, 0.05]}, iters=10
        )
        sched_d = build_schedule(
            {"name": "linear_schedule", "arguments": [0.2, 0.05, 10]}
        )
        for t in (0, 5, 10):
            self.assertAlmostEqual(sched_c(t), sched_d(t), delta=1e-8)

        # Three-arg form matches MLX reference schedule across steps
        sched_e = build_schedule(
            {"name": "linear_schedule", "arguments": [0.3, 0.1, 5]}
        )
        ref_e = opt.schedulers.linear_schedule(0.3, 0.1, 5)
        for t in (0, 1, 4, 5):
            self.assertAlmostEqual(sched_e(t), ref_e(t), delta=1e-8)

    def test_cosine_decay_with_grad_accum_conversion(self):
        # decay_steps should be converted by grad_accumulation_steps; compare to an equivalent schedule
        sched = build_schedule(
            {"name": "cosine_decay", "arguments": [0.1, 20]}, grad_accumulation_steps=2
        )
        ref = build_schedule(
            {"name": "cosine_decay", "arguments": [0.1, 10]}, grad_accumulation_steps=1
        )
        for t in (0, 5, 10):
            self.assertAlmostEqual(sched(t), ref(t), delta=1e-6)

    def test_step_decay_defaults_and_conversion(self):
        # Only lr provided; step_size defaults to iters//10 converted; compare to explicit form
        sched = build_schedule({"name": "step_decay", "arguments": [0.1]}, iters=100)
        ref = build_schedule({"name": "step_decay", "arguments": [0.1, 0.5, 10]})
        for t in (0, 10, 20, 30):
            self.assertAlmostEqual(sched(t), ref(t), delta=1e-6)

    def test_warmup_only_config(self):
        config = {"warmup": 100}
        lr_schedule = build_schedule(config, learning_rate=1e-5)
        self.assertAlmostEqual(lr_schedule(0), 0.0, delta=1e-7)
        self.assertAlmostEqual(lr_schedule(100), 1e-5, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
