"""Regression: exact prompt-cache hits must leave one token for generation."""
import unittest

from mlx_lm.server import _generation_safe_cache_hit


class FakeKV:
    def __init__(self, trimmable=False, size=3):
        self._trimmable = trimmable
        self.size = size

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        n = min(n, self.size)
        self.size -= n
        return n


class FakePromptCache:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def fetch_nearest_cache(self, model, tokens):
        self.calls.append((model, list(tokens)))
        return self.response


class ExactCacheTailTest(unittest.TestCase):
    def test_nontrimmable_exact_uses_prelast(self):
        pre = [FakeKV(False, 2)]
        prompt_cache = FakePromptCache((pre, []))
        cache, rest = _generation_safe_cache_hit(
            prompt_cache, "m", [1, 2, 3], [FakeKV(False, 3)], []
        )
        self.assertIs(cache, pre)
        self.assertEqual(rest, [3])
        self.assertEqual(prompt_cache.calls, [("m", [1, 2])])

    def test_nontrimmable_exact_without_shorter_recomputes(self):
        prompt_cache = FakePromptCache((None, [1, 2]))
        cache, rest = _generation_safe_cache_hit(
            prompt_cache, "m", [1, 2, 3], [FakeKV(False, 3)], []
        )
        self.assertIsNone(cache)
        self.assertEqual(rest, [1, 2, 3])

    def test_trimmable_exact_replays_last_token(self):
        kv = FakeKV(True, 3)
        prompt_cache = FakePromptCache((None, []))
        cache, rest = _generation_safe_cache_hit(
            prompt_cache, "m", [1, 2, 3], [kv], []
        )
        assert cache is not None
        self.assertIs(cache[0], kv)
        self.assertEqual(kv.size, 2)
        self.assertEqual(rest, [3])
        self.assertEqual(prompt_cache.calls, [])

    def test_non_exact_hit_is_unchanged(self):
        cache = [FakeKV(False, 2)]
        prompt_cache = FakePromptCache((None, []))
        got_cache, rest = _generation_safe_cache_hit(
            prompt_cache, "m", [1, 2, 3], cache, [3]
        )
        self.assertIs(got_cache, cache)
        self.assertEqual(rest, [3])
        self.assertEqual(prompt_cache.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
