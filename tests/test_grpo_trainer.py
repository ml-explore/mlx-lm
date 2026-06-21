# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.tuner.grpo_trainer import GRPOArgs, GRPOMetrics, grpo_train
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.utils import load


class TestGRPOTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")
        cls.model.set_dtype(mx.float32)
        # Apply LoRA once for all tests
        linear_to_lora_layers(cls.model, 4, {"rank": 4, "alpha": 8, "dropout": 0.0, "scale": 2.0})

    def test_grpo_train_basic(self):
        """Test that grpo_train runs without error and returns summary."""
        prompts = ["What is 2+2?", "What is 3+3?"]

        def reward_fn(completions, prompt):
            # Simple reward: 1.0 if "4" or "6" in response
            return [1.0 if any(c in comp for c in ["4", "6"]) else 0.0
                    for comp in completions]

        summary = grpo_train(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            reward_fn=reward_fn,
            args=GRPOArgs(
                num_completions=2,
                max_new_tokens=10,
                iters=2,
                learning_rate=1e-4,
                steps_per_save=100,  # don't save during test
            ),
        )

        self.assertIn("total_steps", summary)
        self.assertIn("steps_with_signal", summary)
        self.assertIn("mean_reward", summary)
        self.assertEqual(summary["total_steps"], 2)

    def test_grpo_callback(self):
        """Test that callback is called."""
        from mlx_lm.tuner.grpo_trainer import GRPOCallback

        class TestCallback(GRPOCallback):
            def __init__(self):
                self.steps = []

            def on_step(self, metrics):
                self.steps.append(metrics)

        cb = TestCallback()

        grpo_train(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=["Hello"],
            reward_fn=lambda comps, p: [0.0] * len(comps),
            args=GRPOArgs(num_completions=2, max_new_tokens=5, iters=2, steps_per_save=100),
            callback=cb,
        )

        self.assertEqual(len(cb.steps), 2)
        self.assertIsInstance(cb.steps[0], GRPOMetrics)

    def test_grpo_all_same_reward_skips(self):
        """When all rewards are the same, step is skipped (no gradient signal)."""
        # Always return same reward → no gradient signal
        summary = grpo_train(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=["Test"],
            reward_fn=lambda comps, p: [1.0] * len(comps),
            args=GRPOArgs(num_completions=2, max_new_tokens=5, iters=3, steps_per_save=100),
        )

        self.assertEqual(summary["steps_with_signal"], 0)


if __name__ == "__main__":
    unittest.main()
