# ABOUTME: Verifies paged model cache adapters stream into KVBlockManager.
# ABOUTME: Ensures helper factory builds per-layer adapters without mutation.

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.server_batched.util import ensure_mlx_stub

ensure_mlx_stub()

import mlx.core as mx

import mlx_lm.models.cache as cache_module

PagedKVCacheAdapter = getattr(cache_module, "PagedKVCacheAdapter")
make_paged_prompt_cache = getattr(cache_module, "make_paged_prompt_cache")

try:  # pragma: no cover - optional dependency
    from mlx.nn.paged_kv import KVBlockManager
except Exception:  # pragma: no cover
    KVBlockManager = None


@unittest.skipUnless(KVBlockManager is not None, "MLX paged_kv not available")
class PagedModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.manager = KVBlockManager(
            num_layers=2,
            num_kv_heads=1,
            head_dim=2,
            block_size=2,
            max_blocks=8,
            dtype=mx.float16,
        )

    def test_update_and_fetch_streams_into_manager(self):
        self.manager.new_sequence(seq_id=0, prompt_len=0)
        adapter = PagedKVCacheAdapter(self.manager, seq_id=0, layer_idx=0)
        keys = mx.ones((1, 1, 2, 2), dtype=mx.float16)
        values = mx.ones((1, 1, 2, 2), dtype=mx.float16) * 2

        adapter.update_and_fetch(keys, values)

        tables, ctx_len = self.manager.batch_tables([0])
        self.assertEqual(int(ctx_len[0]), 2)
        blocks_in_use = max(1, math.ceil(int(ctx_len[0]) / self.manager.block_size))
        active_slice = tables[0, :blocks_in_use]
        inactive_slice = tables[0, blocks_in_use:]
        self.assertTrue(bool(mx.all(active_slice >= 0)))
        if inactive_slice.size:
            self.assertTrue(bool(mx.all(inactive_slice == -1)))

    def test_make_paged_prompt_cache_returns_per_layer_adapters(self):
        self.manager.new_sequence(seq_id=1, prompt_len=0)
        caches = make_paged_prompt_cache(self.manager, seq_id=1, num_layers=2)
        self.assertEqual(len(caches), 2)
        self.assertIsInstance(caches[0], PagedKVCacheAdapter)
        caches[0].update_and_fetch(
            mx.ones((1, 1, 1, 2), dtype=mx.float16),
            mx.ones((1, 1, 1, 2), dtype=mx.float16),
        )
        tables, ctx_len = self.manager.batch_tables([1])
        self.assertEqual(int(ctx_len[0]), 1)


if __name__ == "__main__":
    unittest.main()
