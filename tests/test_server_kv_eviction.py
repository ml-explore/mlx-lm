# Copyright © 2024 Apple Inc.

import unittest
from argparse import Namespace

import mlx.core as mx
from test_snapkv_cache import TinyAttnModel

from mlx_lm.models.cache import PositionPreservingKVCache
from mlx_lm.server import ResponseGenerator


def _args(**overrides):
    base = dict(
        kv_eviction="none",
        kv_budget=48,
        kv_window=24,
        kv_sink_tokens=4,
        kv_min_tokens=128,
    )
    base.update(overrides)
    return Namespace(**base)


class _StubGenerator:
    """Exercise ResponseGenerator._maybe_snapkv_compact without a live server."""

    def __init__(self, args):
        self._args = args

    @property
    def cli_args(self):
        return self._args

    _maybe_snapkv_compact = ResponseGenerator._maybe_snapkv_compact


class TestSnapKVServerCompaction(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(self.model.parameters())
        self.prompt = [int(t) for t in mx.random.randint(0, 64, shape=(260,)).tolist()]

    def _fresh_cache(self):
        return self.model.make_cache()

    def test_eviction_off_is_passthrough(self):
        gen = _StubGenerator(_args(kv_eviction="none"))
        cache = self._fresh_cache()
        out_cache, decode = gen._maybe_snapkv_compact(
            self.model, cache, self.prompt, made_fresh=True
        )
        self.assertIs(out_cache, cache)
        self.assertEqual(decode, self.prompt)

    def test_snapkv_compacts_fresh_long_prompt(self):
        gen = _StubGenerator(_args(kv_eviction="snapkv", kv_budget=48))
        cache = self._fresh_cache()
        out_cache, decode = gen._maybe_snapkv_compact(
            self.model, cache, self.prompt, made_fresh=True
        )
        # Decodes from the last prompt token.
        self.assertEqual(decode, self.prompt[-1:])
        # Full-attention layers are compacted, offset preserved at prompt-1.
        for c in out_cache:
            self.assertIsInstance(c, PositionPreservingKVCache)
            self.assertEqual(c.offset, len(self.prompt) - 1)
            self.assertEqual(c.size(), 48)
        # And it actually decodes.
        logits = self.model(mx.array([decode]), cache=out_cache)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 1, 64))

    def test_short_prompt_not_compacted(self):
        gen = _StubGenerator(_args(kv_eviction="snapkv", kv_min_tokens=128))
        short = self.prompt[:100]
        cache = self._fresh_cache()
        out_cache, decode = gen._maybe_snapkv_compact(
            self.model, cache, short, made_fresh=True
        )
        self.assertIs(out_cache, cache)
        self.assertEqual(decode, short)

    def test_partial_hit_not_compacted(self):
        # made_fresh=False means a partial prefix-cache hit; skip compaction.
        gen = _StubGenerator(_args(kv_eviction="snapkv"))
        cache = self._fresh_cache()
        out_cache, decode = gen._maybe_snapkv_compact(
            self.model, cache, self.prompt, made_fresh=False
        )
        self.assertIs(out_cache, cache)
        self.assertEqual(decode, self.prompt)


if __name__ == "__main__":
    unittest.main()
