import unittest

import mlx.core as mx

from mlx_lm.sample_utils import make_sampler

VOCAB = 64
STEPS = 32


def flat_logits():
    return mx.zeros((1, VOCAB))


def step_logits(i):
    # Deterministic per-step logits without touching global RNG state
    return mx.random.normal((1, VOCAB), key=mx.random.key(1000 + i))


def draw(sampler, steps=STEPS, logits_fn=lambda i: flat_logits()):
    return [sampler(logits_fn(i)).item() for i in range(steps)]


class TestKeyedSampler(unittest.TestCase):
    def test_same_seed_identical_sequences(self):
        a = make_sampler(temp=1.0, seed=42)
        b = make_sampler(temp=1.0, seed=42)
        seq_a = draw(a, logits_fn=step_logits)
        seq_b = draw(b, logits_fn=step_logits)
        self.assertEqual(seq_a, seq_b)
        self.assertTrue(all(0 <= t < VOCAB for t in seq_a))

    def test_different_seeds_differ(self):
        a = make_sampler(temp=1.0, seed=0)
        b = make_sampler(temp=1.0, seed=1)
        self.assertNotEqual(draw(a), draw(b))

    def test_seed_none_still_runs(self):
        sampler = make_sampler(temp=1.0)
        token = sampler(flat_logits()).item()
        self.assertTrue(0 <= token < VOCAB)

        argmax_sampler = make_sampler(temp=0.0, seed=None)
        logits = mx.zeros((1, VOCAB)).at[:, 3].add(1.0)
        self.assertEqual(argmax_sampler(logits).item(), 3)

    def test_consecutive_calls_advance_stream(self):
        sampler = make_sampler(temp=1.0, seed=7)
        seq = draw(sampler)
        self.assertGreater(len(set(seq)), 1)

    def test_same_seed_with_full_filter_chain(self):
        kwargs = dict(
            temp=0.8,
            top_p=0.9,
            min_p=0.05,
            top_k=40,
            xtc_probability=0.5,
            xtc_threshold=0.1,
            xtc_special_tokens=[0],
            seed=123,
        )
        seq_a = draw(make_sampler(**kwargs), logits_fn=step_logits)
        seq_b = draw(make_sampler(**kwargs), logits_fn=step_logits)
        self.assertEqual(seq_a, seq_b)

    def test_keyed_stream_independent_of_global_state(self):
        reference = draw(make_sampler(temp=1.0, seed=99), steps=8)

        mx.random.seed(0)
        sampler = make_sampler(temp=1.0, seed=99)
        interleaved = []
        for _ in range(8):
            mx.random.uniform()  # perturb global RNG between draws
            interleaved.append(sampler(flat_logits()).item())
        self.assertEqual(reference, interleaved)


class TestKeyedSampler(unittest.TestCase):
    def test_same_seed_identical_sequences(self):
        a = make_sampler(temp=1.0, seed=42)
        b = make_sampler(temp=1.0, seed=42)
        seq_a = draw(a, logits_fn=step_logits)
        seq_b = draw(b, logits_fn=step_logits)
        self.assertEqual(seq_a, seq_b)
        self.assertTrue(all(0 <= t < VOCAB for t in seq_a))

    def test_different_seeds_differ(self):
        a = make_sampler(temp=1.0, seed=0)
        b = make_sampler(temp=1.0, seed=1)
        self.assertNotEqual(draw(a), draw(b))

    def test_seed_none_still_runs(self):
        sampler = make_sampler(temp=1.0)
        token = sampler(flat_logits()).item()
        self.assertTrue(0 <= token < VOCAB)

        argmax_sampler = make_sampler(temp=0.0, seed=None)
        logits = mx.zeros((1, VOCAB)).at[:, 3].add(1.0)
        self.assertEqual(argmax_sampler(logits).item(), 3)

    def test_consecutive_calls_advance_stream(self):
        sampler = make_sampler(temp=1.0, seed=7)
        seq = draw(sampler)
        self.assertGreater(len(set(seq)), 1)

    def test_same_seed_with_full_filter_chain(self):
        kwargs = dict(
            temp=0.8,
            top_p=0.9,
            min_p=0.05,
            top_k=40,
            xtc_probability=0.5,
            xtc_threshold=0.1,
            xtc_special_tokens=[0],
            seed=123,
        )
        seq_a = draw(make_sampler(**kwargs), logits_fn=step_logits)
        seq_b = draw(make_sampler(**kwargs), logits_fn=step_logits)
        self.assertEqual(seq_a, seq_b)

    def test_keyed_stream_independent_of_global_state(self):
        reference = draw(make_sampler(temp=1.0, seed=99), steps=8)

        mx.random.seed(0)
        sampler = make_sampler(temp=1.0, seed=99)
        interleaved = []
        for _ in range(8):
            mx.random.uniform()  # perturb global RNG between draws
            interleaved.append(sampler(flat_logits()).item())
        self.assertEqual(reference, interleaved)


def make_kv_cache(n_tokens):
    cache = KVCache()
    kv = mx.zeros((1, 1, n_tokens, 4))
    cache.update_and_fetch(kv, kv)
    return [cache]


if __name__ == "__main__":
    unittest.main()
