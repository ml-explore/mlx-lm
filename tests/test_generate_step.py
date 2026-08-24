# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generate_step


class DummyLM(nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.layers = []
        self.vocab_size = vocab_size

    def make_cache(self):
        return []

    def __call__(self, tokens, cache=None, input_embeddings=None):
        seq = tokens if input_embeddings is None else input_embeddings
        return mx.zeros((seq.shape[0], seq.shape[1], self.vocab_size))


class TestGenerateStepLogitsProcessors(unittest.TestCase):
    def test_processor_sees_prefilled_prompt_tokens(self):
        prompt_ids = mx.array([2, 5, 7, 9])
        histories = []

        def record_history(tokens, logits):
            histories.append(tokens.tolist())
            return logits

        list(
            generate_step(
                prompt_ids,
                DummyLM(),
                max_tokens=1,
                prefill_step_size=2,
                logits_processors=[record_history],
            )
        )

        self.assertGreaterEqual(len(histories), 1)
        self.assertEqual(histories[0], [2, 5, 7, 9])


if __name__ == "__main__":
    unittest.main()
