# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.tuner.trainer import iterate_batches


class MockDistributedGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class TestTunerTrainer(unittest.TestCase):
    def test_iterate_batches_ddp(self):
        group = MockDistributedGroup(0, 1)

        def run(rank, size, batch):
            group._rank = rank
            group._size = size

            data = mx.arange(128).reshape(-1, 1).tolist()
            data = [(d, 0) for d in data]

            samples = set()
            for i, (b, l) in enumerate(
                iterate_batches(data, batch, 1, comm_group=group)
            ):
                samples.add(tuple(mx.flatten(b).tolist()))

            ref_batches = mx.arange(128).reshape(-1, batch).tolist()
            for b in ref_batches:
                self.assertTrue(tuple(b[rank::size]) in samples)

        run(0, 1, 4)
        run(0, 1, 8)
        run(0, 2, 8)
        run(1, 2, 8)
        run(0, 4, 8)
        run(1, 4, 8)
        run(2, 4, 8)
        run(3, 4, 8)


from unittest.mock import MagicMock

from mlx_lm.tuner.datasets import ChatDataset, CompletionsDataset, TextDataset


def make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = lambda msgs, **kw: [1, 2, 3, 4, 5]
    tokenizer.eos_token_id = 2
    tokenizer.encode = lambda text: [1, 2, 3]
    return tokenizer


class TestIterateBatchesWithDatasets(unittest.TestCase):
    def test_chat_dataset(self):
        data = [
            {"messages": [{"role": "user", "content": f"msg{i}"}]} for i in range(4)
        ]
        dataset = ChatDataset(data, make_tokenizer())
        batches = iterate_batches(dataset, batch_size=2, max_seq_length=512)
        batch, lengths = next(batches)
        self.assertIsNotNone(batch)

    def test_text_dataset(self):
        data = [{"text": f"hello {i}"} for i in range(4)]
        dataset = TextDataset(data, make_tokenizer())
        batches = iterate_batches(dataset, batch_size=2, max_seq_length=512)
        batch, lengths = next(batches)
        self.assertIsNotNone(batch)

    def test_completions_dataset(self):
        data = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(4)]
        dataset = CompletionsDataset(
            data, make_tokenizer(), "prompt", "completion", False
        )
        batches = iterate_batches(dataset, batch_size=2, max_seq_length=512)
        batch, lengths = next(batches)
        self.assertIsNotNone(batch)


if __name__ == "__main__":
    unittest.main()
