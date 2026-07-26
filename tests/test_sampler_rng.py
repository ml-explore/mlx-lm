# Copyright © 2024 Apple Inc.

"""Per-sampler PRNG streams (fixes: seed silently ignored, identical requests identical output).

Before this change every sampler drew from the implicit global ``mx.random.state``, so
``mlx_lm.server`` returned byte-identical tokens for identical requests at temperature > 0 and the
``seed`` request field had no effect on the output. These tests lock the contract:

  * a seed reproduces the token stream exactly,
  * different seeds diverge,
  * no seed means independent streams per sampler,
  * concurrent/interleaved samplers cannot perturb each other,
  * ``temp == 0`` stays argmax, and the legacy one-argument call signature still works.
"""

import unittest

import mlx.core as mx

from mlx_lm.sample_utils import apply_xtc, categorical_sampling, make_sampler
from mlx_lm.server import _make_sampler


def _logits(vocab=64, seed=0):
    # A flat-ish distribution so sampling genuinely has entropy to draw on.
    mx.random.seed(seed)
    return mx.random.normal((1, vocab))


def _draw(sampler, logprobs, n=24):
    return [int(sampler(logprobs).item()) for _ in range(n)]


class TestSamplerSeeding(unittest.TestCase):
    def test_same_seed_reproduces_stream(self):
        lp = _logits()
        a = _draw(make_sampler(temp=1.0, seed=1234), lp)
        b = _draw(make_sampler(temp=1.0, seed=1234), lp)
        self.assertEqual(a, b)

    def test_different_seeds_diverge(self):
        lp = _logits()
        a = _draw(make_sampler(temp=1.0, seed=1), lp)
        b = _draw(make_sampler(temp=1.0, seed=2), lp)
        self.assertNotEqual(a, b)

    def test_stream_advances_within_a_sampler(self):
        # A single sampler must not return the same token every call.
        lp = _logits()
        draws = _draw(make_sampler(temp=1.0, seed=7), lp, n=32)
        self.assertGreater(len(set(draws)), 1)

    def test_interleaving_does_not_perturb(self):
        # THE concurrency property: interleaving another sampler's draws must not change
        # what this sampler produces. Global-state sampling fails this.
        lp = _logits()
        solo = _draw(make_sampler(temp=1.0, seed=99), lp)

        s = make_sampler(temp=1.0, seed=99)
        other = make_sampler(temp=1.0, seed=1000)
        interleaved = []
        for _ in range(24):
            interleaved.append(int(s(lp).item()))
            other(lp)  # a concurrent generation stealing draws
        self.assertEqual(solo, interleaved)

    def test_seed_survives_the_full_sampler_chain(self):
        # top_p / min_p / top_k each consume a subkey; the chain must stay reproducible.
        lp = _logits()
        kw = dict(temp=1.0, top_p=0.95, min_p=0.02, top_k=40)
        self.assertEqual(
            _draw(make_sampler(seed=5, **kw), lp),
            _draw(make_sampler(seed=5, **kw), lp),
        )
        self.assertNotEqual(
            _draw(make_sampler(seed=5, **kw), lp),
            _draw(make_sampler(seed=6, **kw), lp),
        )

    def test_xtc_path_is_seeded(self):
        lp = _logits()
        kw = dict(temp=1.0, xtc_probability=0.5, xtc_threshold=0.1, xtc_special_tokens=[0])
        self.assertEqual(
            _draw(make_sampler(seed=3, **kw), lp),
            _draw(make_sampler(seed=3, **kw), lp),
        )


class _Args:
    """Minimal stand-in for the server's generation arguments."""

    class _S:
        temperature, top_p, top_k, min_p = 1.0, 0.95, 0, 0.0
        xtc_probability, xtc_threshold = 0.0, 0.1

    def __init__(self, seed=None):
        self.sampling = self._S()
        self.seed = seed


class _Tok:
    eos_token_ids = [0]

    def encode(self, s, **kw):
        return [10]


class TestServerRequestSeeding(unittest.TestCase):
    """The production case: no `seed` in the request body."""

    def test_unseeded_requests_get_independent_streams(self):
        # Two requests with NO seed must not return identical tokens. This is the
        # regression under test — previously every request shared the global state.
        lp = _logits()
        a = _draw(_make_sampler(_Args(seed=None), _Tok()), lp)
        b = _draw(_make_sampler(_Args(seed=None), _Tok()), lp)
        self.assertNotEqual(a, b)

    def test_explicit_request_seed_is_reproducible(self):
        lp = _logits()
        a = _draw(_make_sampler(_Args(seed=4242), _Tok()), lp)
        b = _draw(_make_sampler(_Args(seed=4242), _Tok()), lp)
        self.assertEqual(a, b)

    def test_different_request_seeds_diverge(self):
        lp = _logits()
        a = _draw(_make_sampler(_Args(seed=1), _Tok()), lp)
        b = _draw(_make_sampler(_Args(seed=2), _Tok()), lp)
        self.assertNotEqual(a, b)


class TestBackwardCompatibility(unittest.TestCase):
    def test_temp_zero_is_argmax(self):
        lp = _logits()
        sampler = make_sampler(temp=0.0, seed=42)
        self.assertEqual(int(sampler(lp).item()), int(mx.argmax(lp, axis=-1).item()))

    def test_temp_zero_accepts_a_key_argument(self):
        # The temp==0 fast path must accept the same call shape as the real sampler.
        lp = _logits()
        make_sampler(temp=0.0)(lp, None)

    def test_legacy_single_argument_call_still_works(self):
        # Third-party callers typed as Callable[[mx.array], mx.array] must keep working.
        lp = _logits()
        self.assertIsNotNone(make_sampler(temp=1.0)(lp))

    def test_unseeded_derives_from_global_state(self):
        # With no explicit seed the stream is DERIVED from the global RNG at
        # construction, so mx.random.seed(...) set beforehand still pins the result.
        # The sampler must be rebuilt after reseeding — derivation happens once.
        lp = _logits()
        mx.random.seed(77)
        a = _draw(make_sampler(temp=1.0), lp, n=8)
        mx.random.seed(77)
        b = _draw(make_sampler(temp=1.0), lp, n=8)
        self.assertEqual(a, b)

    def test_unseeded_samplers_differ_without_reseeding(self):
        # Two samplers built back to back (no global reseed) must get independent
        # streams — this is the server's no-seed request path.
        lp = _logits()
        mx.random.seed(5)
        a = _draw(make_sampler(temp=1.0), lp, n=12)
        b = _draw(make_sampler(temp=1.0), lp, n=12)
        self.assertNotEqual(a, b)


class TestCompiledFunctionsAcceptNone(unittest.TestCase):
    """The sampler never passes None into the mx.compile'd helpers, but external
    callers using the pre-existing signatures still can. These lock that path."""

    def test_categorical_sampling_without_key(self):
        lp = _logits()
        self.assertIsNotNone(categorical_sampling(lp, 1.0))

    def test_apply_xtc_without_key(self):
        lp = _logits()
        self.assertEqual(apply_xtc(lp, 0.5, 0.1, [0]).shape, lp.shape)


if __name__ == "__main__":
    unittest.main()
