import unittest

import mlx.core as mx

from mlx_lm.models.cache import KVCache, LRUPromptCache
from mlx_lm.server import ResponseGenerator


def make_kv_cache(n_tokens):
    cache = KVCache()
    kv = mx.zeros((1, 1, n_tokens, 4))
    cache.update_and_fetch(kv, kv)
    return [cache]


class UntrimmableCache(KVCache):
    def is_trimmable(self):
        return False


class TestExactHitPromptCacheGuard(unittest.TestCase):
    def _generator(self):
        gen = ResponseGenerator.__new__(ResponseGenerator)
        gen.prompt_cache = LRUPromptCache()
        return gen

    def test_exact_hit_leaves_one_token(self):
        gen = self._generator()
        model = ("m", None, None)
        tokens = list(range(10))
        gen.prompt_cache.insert_cache(model, tokens, make_kv_cache(10))

        cache, rest = gen._fetch_prompt_cache(model, tokens)
        self.assertEqual(rest, [tokens[-1]])
        self.assertEqual(cache[0].offset, 9)

    def test_prefix_hit_unchanged(self):
        gen = self._generator()
        model = ("m", None, None)
        tokens = list(range(10))
        gen.prompt_cache.insert_cache(model, tokens, make_kv_cache(10))

        cache, rest = gen._fetch_prompt_cache(model, tokens + [99])
        self.assertEqual(rest, [99])
        self.assertEqual(cache[0].offset, 10)

    def test_untrimmable_exact_hit_recomputes(self):
        gen = self._generator()
        model = ("m", None, None)
        tokens = list(range(10))
        untrimmable = UntrimmableCache()
        kv = mx.zeros((1, 1, 10, 4))
        untrimmable.update_and_fetch(kv, kv)
        gen.prompt_cache.insert_cache(model, tokens, [untrimmable])

        cache, rest = gen._fetch_prompt_cache(model, tokens)
        self.assertIsNone(cache)
        self.assertEqual(rest, tokens)

    def test_miss_unchanged(self):
        gen = self._generator()
        cache, rest = gen._fetch_prompt_cache(("m", None, None), [1, 2, 3])
        self.assertIsNone(cache)
        self.assertEqual(rest, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
