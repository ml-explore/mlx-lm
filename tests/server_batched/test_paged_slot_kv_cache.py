# ABOUTME: Validates paged slot KV cache utilities against KVBlockManager.
# ABOUTME: Ensures batch view exposes block tables/context lengths for decode.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import unittest

import mlx.core as mx

try:
    from mlx.nn.paged_kv import KVBlockManager
except Exception:  # pragma: no cover
    KVBlockManager = None

from mlx_lm.server_batched.paged_slot_kv_cache import (
    PagedBatchView,
    PagedSlotKVCache,
    PrefixCache,
)

KV_MANAGER_AVAILABLE = KVBlockManager is not None and not getattr(
    KVBlockManager, "IS_STUB", False
)


@unittest.skipUnless(KV_MANAGER_AVAILABLE, "KVBlockManager unavailable")
class PagedSlotKVCacheTests(unittest.TestCase):
    def setUp(self):
        self.manager = KVBlockManager(
            num_layers=2,
            num_kv_heads=1,
            head_dim=4,
            block_size=2,
            max_blocks=16,
            dtype=mx.float16,
        )
        self.cache = PagedSlotKVCache(self.manager, max_active=8)

    def test_register_and_batch_tables(self):
        seq_id = self.cache.register("req-1", prompt_len=3)
        self.assertEqual(self.cache.sequence_id("req-1"), seq_id)

        k_chunk = mx.reshape(mx.arange(8, dtype=mx.float16), (1, 2, 4))
        v_chunk = k_chunk + 100
        self.cache.write_prefill(
            seq_id, layer_idx=0, k_chunk=k_chunk, v_chunk=v_chunk, start_pos=0
        )

        tables, ctx = self.cache.batch_tables_from_requests(["req-1"])
        self.assertEqual(int(ctx[0]), 3)
        self.assertGreaterEqual(int(tables[0, 0]), 0)

    def test_begin_prefill_returns_handle(self):
        seq_id = self.cache.register("req-handle", prompt_len=1)
        handle = self.cache.begin_prefill("req-handle", chunk_len=2)
        self.assertEqual(handle.view.size, 1)
        self.assertEqual(int(handle.view.context_lens[0].item()), 3)
        self.assertTrue(
            mx.array_equal(handle.view.prefill_base_lens, mx.array([1], dtype=mx.int32))
        )
        handle.commit()
        _, ctx = self.cache.batch_tables_from_requests(["req-handle"])
        self.assertEqual(int(ctx[0]), 3)

    def test_begin_prefill_many_returns_batch_handle(self):
        reqs = []
        for idx in range(3):
            req_id = f"req-batch-{idx}"
            self.cache.register(req_id, prompt_len=idx + 1)
            reqs.append(req_id)
        handle = self.cache.begin_prefill_many(reqs, [1, 2, 3])
        self.assertEqual(handle.view.size, len(reqs))
        base_lens = handle.view.prefill_base_lens.tolist()
        self.assertEqual(base_lens, [1, 2, 3])
        handle.commit()
        _, ctx = self.cache.batch_tables_from_requests(reqs)
        self.assertEqual([int(v) for v in ctx.tolist()], [2, 4, 6])

    def test_release_clears_sequence(self):
        self.cache.register("req-x", prompt_len=1)
        self.cache.release("req-x")
        with self.assertRaises(KeyError):
            self.cache.sequence_id("req-x")


@unittest.skipUnless(KV_MANAGER_AVAILABLE, "KVBlockManager unavailable")
class PagedBatchViewTests(unittest.TestCase):
    def setUp(self):
        manager = KVBlockManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            block_size=2,
            max_blocks=8,
            dtype=mx.float16,
        )
        self.manager = manager
        self.cache = PagedSlotKVCache(manager, max_active=4)
        self.seq_id = self.cache.register("req-1", prompt_len=2)
        k_chunk = mx.ones((1, 2, 2), dtype=mx.float16)
        v_chunk = mx.ones((1, 2, 2), dtype=mx.float16)
        self.cache.write_prefill(
            self.seq_id, layer_idx=0, k_chunk=k_chunk, v_chunk=v_chunk, start_pos=0
        )

    def test_args_for_layer(self):
        view = PagedBatchView(self.manager, seq_ids=[self.seq_id], kv_head_mapping=None)
        args = view.args_for_layer(0)
        self.assertEqual(len(args), 7)
        self.assertEqual(view.size, 1)
        self.assertEqual(args[0].shape[0], self.manager.num_kv_heads)
        self.assertIsInstance(args[-1], dict)


class PrefixCacheTests(unittest.TestCase):
    def test_record_and_reuse(self):
        class ManagerStub:
            def __init__(self):
                self.reused = None

            def snapshot_blocks(self, seq_id, seq_len):
                return [0, 1]

            def reuse_prefix(self, seq_id, block_ids, seq_len):
                self.reused = (seq_id, tuple(block_ids), seq_len)

        mgr = ManagerStub()
        cache = PrefixCache(mgr, block_size=2, max_entries=2)
        key = b"prefix"
        cache.record(key, seq_id=0, seq_len=4)
        reused = cache.try_reuse(key, seq_id=1, seq_len=4)
        self.assertEqual(reused, 4)
        self.assertEqual(mgr.reused, (1, (0, 1), 4))
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1.0)
        self.assertEqual(stats["lookups"], 1.0)
        self.assertEqual(stats["tokens_reused"], 4.0)

    def test_partial_reuse_rejects_non_matching_length(self):
        class ManagerStub:
            def snapshot_blocks(self, seq_id, seq_len):
                return [0]

            def reuse_prefix(self, seq_id, block_ids, seq_len):
                raise AssertionError("should not reuse")

        mgr = ManagerStub()
        cache = PrefixCache(mgr, block_size=2, max_entries=4)
        key = b"prefix"
        cache.record(key, seq_id=0, seq_len=4)
        reused = cache.try_reuse(key, seq_id=1, seq_len=2)
        self.assertEqual(reused, 0)
        stats = cache.stats()
        self.assertEqual(stats["hits"], 0.0)
        self.assertEqual(stats["lookups"], 1.0)
        self.assertEqual(stats["tokens_reused"], 0.0)

    def test_record_many_persists_all_prefixes(self):
        class ManagerStub:
            def __init__(self):
                self.snapshot_calls = []
                self.reuse_calls = []

            def snapshot_blocks(self, seq_id, seq_len):
                self.snapshot_calls.append((seq_id, seq_len))
                return list(range(seq_len // 2))

            def reuse_prefix(self, seq_id, block_ids, seq_len):
                self.reuse_calls.append((seq_id, tuple(block_ids), seq_len))

        mgr = ManagerStub()
        cache = PrefixCache(mgr, block_size=2, max_entries=4)
        prefixes = [(2, b"a"), (4, b"b")]
        cache.record_many(prefixes, seq_id=7)
        self.assertEqual(len(mgr.snapshot_calls), 2)
        reused = cache.try_reuse(b"b", seq_id=8, seq_len=4)
        self.assertEqual(reused, 4)
        self.assertEqual(mgr.reuse_calls[-1], (8, (0, 1), 4))


if __name__ == "__main__":
    unittest.main()
