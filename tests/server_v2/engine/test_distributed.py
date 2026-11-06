# Copyright © 2026 Apple Inc.

import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.server_v2.engine import core, distributed


class TestDistributed(unittest.TestCase):

    def test_share(self):
        commands = [core._Cancel(3), core._Close()]
        sent = []

        def all_sum(value, *, group):
            # Rank 1 sends only zeros, so a sum is what rank 0 sent.
            if group.rank() == 0:
                sent.append(mx.array(value))
                return sent[-1]
            return sent.pop(0) + value

        rank_0, rank_1 = mock.Mock(rank=lambda: 0), mock.Mock(rank=lambda: 1)
        with mock.patch.object(mx.distributed, "all_sum", all_sum):
            self.assertEqual(distributed.share(rank_0, commands), commands)
            self.assertEqual(distributed.share(rank_1, []), commands)


if __name__ == "__main__":
    unittest.main()
