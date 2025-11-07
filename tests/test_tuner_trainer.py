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


if __name__ == "__main__":
    unittest.main()
