# Copyright © 2024 Apple Inc.

"""Regression tests for #1662.

A cache field the sampled-token graph never reads (e.g. K == V windowed
attention that discards the ``values`` half, as in ``deepseek_v4``)
accumulates one unevaluated in-place update per decode step. Nothing
downstream forces it to materialize, so the lazy graph grows without bound
and eventually exceeds the runtime's Metal buffer-count limit.

Model-free, mirroring the issue's own minimal repro and the testing style of
the accepted fix for the same bug class in #1632 (``mx.export_to_dot`` +
edge count on the field that must not keep accumulating).
"""

import io
import unittest

import mlx.core as mx

from mlx_lm.generate import (
    CACHE_STATE_EVAL_INTERVAL,
    GenerationBatch,
    StopSequenceMatcher,
    generate_step,
)
from mlx_lm.sample_utils import make_sampler


class _EchoCache:
    """Keys are read by the model below; values are written every step and
    never read back -- the exact shape of the leak in #1662, without
    needing deepseek_v4 (or any other model) to reproduce it.
    """

    def __init__(self):
        self.keys = mx.zeros((1, 1, 1, 8), mx.float32)
        self.values = mx.zeros((1, 1, 1, 8), mx.float32)
        self.offset = 0

    def update_and_fetch(self, keys, values):
        self.keys = mx.concatenate([self.keys, keys], axis=2)[..., -8:, :]
        self.values = mx.concatenate([self.values, values], axis=2)[..., -8:, :]
        self.offset += 1
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    def to_quantized(self, **kwargs):
        return self

    def extract(self, idx):
        return self

    def filter(self, keep):
        pass


class _LeakyModel:
    """Writes ``cache.values`` every step and never reads it back, same
    shape as deepseek_v4's attention (K == V, values half discarded)."""

    def __call__(self, tokens, cache=None):
        row = mx.ones((1, 1, 1, 8), mx.float32) * tokens.sum()
        keys, _ = cache[0].update_and_fetch(row, row)
        return mx.broadcast_to(
            keys.sum()[None, None, None], (tokens.shape[0], tokens.shape[1], 4)
        )


def _dead_chain_edges(array):
    f = io.StringIO()
    mx.export_to_dot(f, array)
    f.seek(0)
    return f.read().count("->")


class TestDecodeCacheStateEval(unittest.TestCase):
    def _run_generate_step(self, n_steps):
        cache = [_EchoCache()]
        for _ in generate_step(
            mx.array([0]), _LeakyModel(), max_tokens=n_steps, prompt_cache=cache
        ):
            pass
        return _dead_chain_edges(cache[0].values)

    def test_generate_step_bounds_cache_graph(self):
        # Compare two run lengths rather than assert an absolute edge count:
        # mx.export_to_dot's edge-per-op cost is an implementation detail,
        # but "does the graph keep growing with total steps" is not. Without
        # the periodic cache-state eval, values never gets materialized and
        # the chain grows linearly with n_steps; with it, it is bounded by
        # CACHE_STATE_EVAL_INTERVAL regardless of how long generation runs.
        short = self._run_generate_step(CACHE_STATE_EVAL_INTERVAL * 2)
        long = self._run_generate_step(CACHE_STATE_EVAL_INTERVAL * 8)
        self.assertLess(long, short * 2)

    def _run_generation_batch(self, n_steps):
        cache = [_EchoCache()]
        sole_cache = cache[0]  # batch.filter() clears the list in place
        batch = GenerationBatch(
            model=_LeakyModel(),
            uids=[0],
            inputs=mx.array([0]),
            prompt_cache=cache,
            tokens=[[0]],
            samplers=[None],
            fallback_sampler=make_sampler(temp=0.0),
            logits_processors=[None],
            stop_matchers=[StopSequenceMatcher()],
            max_tokens=[n_steps],
        )
        while batch.next():
            pass
        return _dead_chain_edges(sole_cache.values)

    def test_generation_batch_bounds_cache_graph(self):
        short = self._run_generation_batch(CACHE_STATE_EVAL_INTERVAL * 2)
        long = self._run_generation_batch(CACHE_STATE_EVAL_INTERVAL * 8)
        self.assertLess(long, short * 2)


if __name__ == "__main__":
    unittest.main()
