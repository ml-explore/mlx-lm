# Copyright © 2025 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm.tuner.datasets import PreferenceDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text):
        """Simple encoding: return list of character codes."""
        return [ord(c) for c in text[:10]]  # Limit to 10 chars for testing

    def apply_chat_template(self, messages):
        """Mock chat template application."""
        if not messages:
            return []

        # Simple concatenation of message contents
        result = []
        for message in messages:
            content = message.get("content", "")
            result.extend([ord(c) for c in content[:5]])  # Limit for testing

        return result


class TestPreferenceDataset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MockTokenizer()

        # Sample preference data
        self.sample_data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "AI is bad.",
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "chosen": "Hi there! How can I help?",
                "rejected": "Hi.",
            },
        ]

    def test_preference_dataset_creation(self):
        """Test basic creation of PreferenceDataset."""
        dataset = PreferenceDataset(
            data=self.sample_data,
            tokenizer=self.tokenizer,
            max_length=50,
        )

        self.assertEqual(len(dataset), 2)

    def test_preference_dataset_process_prompt_format(self):
        """Test processing of prompt+chosen/rejected format."""
        dataset = PreferenceDataset(
            data=[self.sample_data[0]],
            tokenizer=self.tokenizer,
            max_length=50,
        )

        processed = dataset.process(self.sample_data[0])

        # Check required keys
        required_keys = [
            "chosen_tokens",
            "rejected_tokens",
            "chosen_length",
            "rejected_length",
            "prompt_length",
        ]
        for key in required_keys:
            self.assertIn(key, processed)

        # Check that tokens are lists
        self.assertIsInstance(processed["chosen_tokens"], list)
        self.assertIsInstance(processed["rejected_tokens"], list)

        # Check that lengths are integers
        self.assertIsInstance(processed["chosen_length"], int)
        self.assertIsInstance(processed["rejected_length"], int)
        self.assertIsInstance(processed["prompt_length"], int)

    def test_preference_dataset_process_messages_format(self):
        """Test processing of messages+chosen/rejected format."""
        dataset = PreferenceDataset(
            data=[self.sample_data[1]],
            tokenizer=self.tokenizer,
            max_length=50,
        )

        processed = dataset.process(self.sample_data[1])

        # Check required keys
        required_keys = [
            "chosen_tokens",
            "rejected_tokens",
            "chosen_length",
            "rejected_length",
            "prompt_length",
        ]
        for key in required_keys:
            self.assertIn(key, processed)

    def test_preference_dataset_eos_token_handling(self):
        """Test that EOS tokens are properly added."""
        dataset = PreferenceDataset(
            data=[self.sample_data[0]],
            tokenizer=self.tokenizer,
            max_length=50,
        )

        processed = dataset.process(self.sample_data[0])

        # Check that sequences end with EOS token
        self.assertEqual(processed["chosen_tokens"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(processed["rejected_tokens"][-1], self.tokenizer.eos_token_id)

    def test_preference_dataset_max_length_truncation(self):
        """Test that sequences are truncated to max_length."""
        short_max_length = 5
        dataset = PreferenceDataset(
            data=[self.sample_data[0]],
            tokenizer=self.tokenizer,
            max_length=short_max_length,
        )

        processed = dataset.process(self.sample_data[0])

        # Check that sequences don't exceed max_length
        self.assertLessEqual(len(processed["chosen_tokens"]), short_max_length)
        self.assertLessEqual(len(processed["rejected_tokens"]), short_max_length)

        # Check that truncated sequences still end with EOS
        if len(processed["chosen_tokens"]) == short_max_length:
            self.assertEqual(
                processed["chosen_tokens"][-1], self.tokenizer.eos_token_id
            )
        if len(processed["rejected_tokens"]) == short_max_length:
            self.assertEqual(
                processed["rejected_tokens"][-1], self.tokenizer.eos_token_id
            )

    def test_preference_dataset_indexing(self):
        """Test dataset indexing."""
        dataset = PreferenceDataset(
            data=self.sample_data,
            tokenizer=self.tokenizer,
            max_length=50,
        )

        # Test valid indexing
        item0 = dataset[0]
        item1 = dataset[1]

        self.assertEqual(item0, self.sample_data[0])
        self.assertEqual(item1, self.sample_data[1])

        # Test out of bounds
        with self.assertRaises(IndexError):
            dataset[10]

    def test_preference_dataset_custom_keys(self):
        """Test dataset with custom key names."""
        custom_data = [
            {
                "question": "What is AI?",
                "good_answer": "AI is artificial intelligence.",
                "bad_answer": "AI is bad.",
            }
        ]

        dataset = PreferenceDataset(
            data=custom_data,
            tokenizer=self.tokenizer,
            max_length=50,
            prompt_key="question",
            chosen_key="good_answer",
            rejected_key="bad_answer",
        )

        processed = dataset.process(custom_data[0])

        # Should still work with custom keys
        self.assertIn("chosen_tokens", processed)
        self.assertIn("rejected_tokens", processed)


class TestDPOIntegration(unittest.TestCase):
    """Integration tests for DPO functionality."""

    def test_dpo_training_setup(self):
        """Test that DPO training components can be imported and initialized."""
        from mlx_lm.tuner.trainer import dpo_iterate_batches, dpo_loss_fn, train_dpo

        # Test that functions are callable
        self.assertTrue(callable(train_dpo))
        self.assertTrue(callable(dpo_loss_fn))
        self.assertTrue(callable(dpo_iterate_batches))

    def test_dpo_cli_entry_point(self):
        """Test that DPO is registered as a CLI subcommand."""
        import mlx_lm.dpo

        # Test that the module has the required functions
        self.assertTrue(hasattr(mlx_lm.dpo, "main"))
        self.assertTrue(hasattr(mlx_lm.dpo, "build_parser"))
        self.assertTrue(hasattr(mlx_lm.dpo, "run"))


if __name__ == "__main__":
    unittest.main()
