import copy
import gc
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.generate import BatchGenerator, _eval_cache_state
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.paged_cache import (
    BatchPagedKVCache,
    BlockTable,
    InvalidPageReferenceError,
    KVBlockPool,
    PageAllocationError,
    PageAllocator,
    PagedKVCache,
    QwenHybridPagedKVCacheManager,
    RequestCapacityError,
    StaleBlockTableError,
)
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs


class TestPageAllocator(unittest.TestCase):
    def test_lifecycle_and_conservation(self):
        allocator = PageAllocator(4)
        pages = allocator.allocate(3)
        allocator.retain(pages[:1])

        stats = allocator.stats()
        self.assertEqual(stats.capacity_pages, 4)
        self.assertEqual(stats.free_pages, 1)
        self.assertEqual(stats.live_pages, 3)
        self.assertEqual(stats.shared_pages, 1)
        self.assertEqual(stats.references, 4)

        allocator.release(pages[:1])
        allocator.release(pages)
        stats = allocator.stats()
        self.assertEqual(stats.free_pages + stats.live_pages, stats.capacity_pages)
        self.assertEqual(stats.free_pages, 4)
        self.assertEqual(stats.references, 0)

    def test_exhaustion_is_atomic(self):
        allocator = PageAllocator(2)
        page = allocator.allocate()
        before = allocator.stats()
        with self.assertRaises(PageAllocationError) as raised:
            allocator.allocate(2)
        self.assertEqual(raised.exception.requested_pages, 2)
        self.assertEqual(raised.exception.available_pages, 1)
        self.assertEqual(raised.exception.shortfall_pages, 1)
        self.assertIsNone(raised.exception.pool_generation)
        self.assertEqual(allocator.stats(), before)
        allocator.release(page)

    def test_rejects_invalid_release(self):
        allocator = PageAllocator(2)
        page = allocator.allocate()
        allocator.release(page)
        with self.assertRaises(InvalidPageReferenceError):
            allocator.release(page)
        with self.assertRaises(InvalidPageReferenceError):
            allocator.retain((3,))
        with self.assertRaises(InvalidPageReferenceError):
            allocator.retain((0, 0))

    def test_one_million_operations_conserve_capacity(self):
        allocator = PageAllocator(17)
        for _ in range(1_000_000):
            pages = allocator.allocate(7)
            allocator.retain(pages[:3])
            allocator.release(pages[:3])
            allocator.release(pages)
        stats = allocator.stats()
        self.assertEqual(stats.free_pages, stats.capacity_pages)
        self.assertEqual(stats.live_pages, 0)
        self.assertEqual(stats.references, 0)


class TestPagedKVCache(unittest.TestCase):
    def make_pool(self, capacity_pages=8, page_size=4):
        return KVBlockPool(
            capacity_pages=capacity_pages,
            page_size=page_size,
            num_kv_heads=2,
            key_head_dim=3,
            value_head_dim=5,
            dtype=mx.float32,
        )

    def make_values(self, start, length):
        keys = mx.arange(start, start + 2 * length * 3, dtype=mx.float32).reshape(
            1, 2, length, 3
        )
        values = mx.arange(
            start + 1000, start + 1000 + 2 * length * 5, dtype=mx.float32
        ).reshape(1, 2, length, 5)
        return keys, values

    def test_append_and_reference_gather(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool, sequence_id="request-a")
        keys_a, values_a = self.make_values(0, 3)
        keys_b, values_b = self.make_values(100, 6)

        cache.update_and_fetch(keys_a, values_a)
        keys, values = cache.update_and_fetch(keys_b, values_b)

        self.assertEqual(cache.size(), 9)
        self.assertEqual(len(cache.block_table.page_ids), 3)
        self.assertTrue(mx.array_equal(keys, mx.concatenate([keys_a, keys_b], axis=2)))
        self.assertTrue(
            mx.array_equal(values, mx.concatenate([values_a, values_b], axis=2))
        )
        pool.validate_table(cache.block_table)

    def test_append_inside_tail_page_skips_allocator(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool, sequence_id="request-a")
        keys, values = self.make_values(0, 2)
        cache._append(keys, values)

        with patch.object(pool, "allocate_for", wraps=pool.allocate_for) as allocate:
            keys, values = self.make_values(100, 1)
            cache._append(keys, values)

        allocate.assert_not_called()
        self.assertEqual(cache.size(), 3)

    def test_device_page_ids_are_reused_until_table_changes(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool, sequence_id="request-a")
        keys, values = self.make_values(0, 2)
        cache._append(keys, values)
        first = cache.device_page_ids()

        keys, values = self.make_values(100, 1)
        cache._append(keys, values)
        self.assertIs(cache.device_page_ids(), first)

        keys, values = self.make_values(200, 2)
        cache._append(keys, values)
        self.assertIsNot(cache.device_page_ids(), first)

    def test_eval_state_does_not_materialize_contiguous_kv(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool, sequence_id="request-a")
        keys, values = self.make_values(0, 3)
        cache._append(keys, values)

        with patch.object(cache, "gather", side_effect=AssertionError):
            _eval_cache_state([cache])

    def test_append_rejects_under_reserved_growth_before_mutation(self):
        pool = self.make_pool(capacity_pages=2)
        pool.reserve("request", 1)
        cache = PagedKVCache(
            pool,
            sequence_id="request",
            reservation_id="request",
        )
        before = cache.block_table
        keys, values = self.make_values(0, 5)

        with self.assertRaises(PageAllocationError):
            cache.update_and_fetch(keys, values)

        self.assertEqual(cache.block_table, before)
        self.assertEqual(pool.reserved_pages, 1)
        self.assertEqual(pool.stats().live_pages, 1)
        pool.cancel_reservation("request")
        cache.release()

    def test_prefix_fork_uses_tail_copy_on_write(self):
        pool = self.make_pool()
        parent = PagedKVCache(pool, sequence_id="parent")
        keys, values = self.make_values(0, 6)
        parent.update_and_fetch(keys, values)
        child = parent.fork(sequence_id="child")

        shared_table = parent.block_table
        self.assertEqual(child.block_table.page_ids, shared_table.page_ids)
        self.assertEqual(pool.stats().shared_pages, 2)

        child_keys, child_values = self.make_values(500, 1)
        child.update_and_fetch(child_keys, child_values)

        self.assertEqual(parent.block_table.page_ids[0], child.block_table.page_ids[0])
        self.assertNotEqual(
            parent.block_table.page_ids[1], child.block_table.page_ids[1]
        )
        self.assertTrue(mx.array_equal(parent.gather()[0], keys))
        self.assertTrue(
            mx.array_equal(
                child.gather()[0], mx.concatenate([keys, child_keys], axis=2)
            )
        )

        child.release()
        parent.release()
        self.assertEqual(pool.stats().free_pages, pool.capacity_pages)

    def test_deepcopy_shares_pages(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool)
        cache.update_and_fetch(*self.make_values(0, 4))
        cloned = copy.deepcopy(cache)
        self.assertEqual(cloned.block_table.page_ids, cache.block_table.page_ids)
        self.assertEqual(pool.stats().shared_pages, 1)
        cloned.release()
        cache.release()

    def test_trim_releases_unused_pages(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool)
        keys, values = self.make_values(0, 9)
        cache.update_and_fetch(keys, values)

        self.assertEqual(cache.trim(5), 5)
        self.assertEqual(cache.size(), 4)
        self.assertEqual(len(cache.block_table.page_ids), 1)
        self.assertTrue(mx.array_equal(cache.gather()[0], keys[..., :4, :]))
        self.assertEqual(pool.stats().live_pages, 1)

    def test_release_is_idempotent(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool)
        cache.update_and_fetch(*self.make_values(0, 1))
        self.assertTrue(cache.release())
        self.assertFalse(cache.release())
        with self.assertRaises(InvalidPageReferenceError):
            cache.gather()

    def test_pool_exhaustion_preserves_cache(self):
        pool = self.make_pool(capacity_pages=1)
        cache = PagedKVCache(pool)
        keys, values = self.make_values(0, 4)
        cache.update_and_fetch(keys, values)
        before = cache.block_table

        with self.assertRaises(PageAllocationError):
            cache.update_and_fetch(*self.make_values(100, 1))

        self.assertEqual(cache.block_table, before)
        self.assertTrue(mx.array_equal(cache.gather()[0], keys))

    def test_stale_and_malformed_block_tables_are_rejected(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool)
        cache.update_and_fetch(*self.make_values(0, 4))
        table = cache.block_table

        with self.assertRaises(StaleBlockTableError):
            pool.validate_table(
                replace(table, pool_generation=table.pool_generation + 1)
            )
        with self.assertRaises(StaleBlockTableError):
            pool.validate_table(replace(table, page_size=table.page_size + 1))
        with self.assertRaises(InvalidPageReferenceError):
            pool.validate_table(
                BlockTable(
                    pool_generation=table.pool_generation,
                    page_size=table.page_size,
                    page_ids=table.page_ids,
                    num_tokens=table.page_size + 1,
                )
            )
        with self.assertRaises(InvalidPageReferenceError):
            pool.validate_table(
                replace(table, page_ids=table.page_ids * 2, num_tokens=8)
            )

    def test_rejects_incompatible_updates(self):
        pool = self.make_pool()
        cache = PagedKVCache(pool)
        with self.assertRaises(ValueError):
            cache.update_and_fetch(
                mx.zeros((2, 2, 1, 3), dtype=mx.float32),
                mx.zeros((2, 2, 1, 5), dtype=mx.float32),
            )
        with self.assertRaises(ValueError):
            cache.update_and_fetch(
                mx.zeros((1, 2, 1, 3), dtype=mx.float16),
                mx.zeros((1, 2, 1, 5), dtype=mx.float16),
            )


class TestBatchPagedKVCache(unittest.TestCase):
    def make_pool(self):
        return KVBlockPool(
            capacity_pages=16,
            page_size=4,
            num_kv_heads=2,
            key_head_dim=3,
            value_head_dim=3,
            dtype=mx.float32,
        )

    def make_values(self, batch_size, length, start=0):
        values = mx.arange(
            start,
            start + batch_size * 2 * length * 3,
            dtype=mx.float32,
        ).reshape(batch_size, 2, length, 3)
        return values, values + 1000

    def test_merge_append_filter_extract_without_physical_rebatch(self):
        pool = self.make_pool()
        row_a = PagedKVCache(pool, sequence_id="a")
        row_b = PagedKVCache(pool, sequence_id="b")
        row_a.update_and_fetch(*self.make_values(1, 3))
        row_b.update_and_fetch(*self.make_values(1, 1, start=100))
        batch = BatchPagedKVCache.merge([row_a, row_b])
        row_a.release()
        row_b.release()

        page_ids_before = [cache.block_table.page_ids for cache in batch.caches]
        keys, values = self.make_values(2, 1, start=200)
        gathered_keys, _ = batch.update_and_fetch(keys, values)
        self.assertEqual(gathered_keys.shape, (2, 2, 4, 3))
        self.assertEqual(batch.caches[0].block_table.page_ids, page_ids_before[0])
        self.assertEqual(batch.caches[1].block_table.page_ids, page_ids_before[1])

        extracted = batch.extract(1)
        batch.filter([0])
        self.assertEqual(len(batch.caches), 1)
        self.assertEqual(extracted.size(), 2)
        extracted.release()
        batch.release()
        self.assertEqual(pool.stats().live_pages, 0)

    def test_right_padded_prefill_only_commits_real_tokens(self):
        pool = self.make_pool()
        batch = BatchPagedKVCache(
            [PagedKVCache(pool, sequence_id="a"), PagedKVCache(pool, sequence_id="b")]
        )
        batch.prepare(lengths=[5, 2], right_padding=[0, 3])
        keys_a, values_a = self.make_values(2, 3)
        keys_b, values_b = self.make_values(2, 2, start=100)
        batch.update_and_fetch(keys_a, values_a)
        batch.update_and_fetch(keys_b, values_b)
        batch.finalize()

        self.assertEqual([cache.size() for cache in batch.caches], [5, 2])
        gathered, _ = batch.gather()
        self.assertEqual(gathered.shape, (2, 2, 5, 3))
        self.assertTrue(mx.all(gathered[1, :, :3, :] == 0))
        batch.release()

    def test_extend_transfers_page_ownership(self):
        pool = self.make_pool()
        left = BatchPagedKVCache([PagedKVCache(pool, sequence_id="a")])
        right = BatchPagedKVCache([PagedKVCache(pool, sequence_id="b")])
        left.extend(right)
        self.assertEqual(len(left.caches), 2)
        self.assertFalse(right.release())
        left.release()


class TestQwenHybridPagedKVCacheManager(unittest.TestCase):
    def make_model(self, model_type="qwen3_5"):
        mx.random.seed(0)
        args = TextModelArgs(
            model_type=model_type,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            vocab_size=128,
            max_position_embeddings=128,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            full_attention_interval=2,
            num_experts=2 if model_type == "qwen3_5_moe" else 0,
            num_experts_per_tok=1 if model_type == "qwen3_5_moe" else 0,
            moe_intermediate_size=16 if model_type == "qwen3_5_moe" else 0,
            shared_expert_intermediate_size=(16 if model_type == "qwen3_5_moe" else 0),
        )
        return TextModel(args)

    def make_target_stub(self, model_type):
        specs = {
            "qwen3_5_moe": (40, 16, 2, 256, 4),
            "qwen3_5": (64, 24, 4, 256, 4),
        }
        num_layers, num_query_heads, num_kv_heads, head_dim, interval = specs[
            model_type
        ]
        args = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_query_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            full_attention_interval=interval,
        )
        layers = []
        for layer_index in range(num_layers):
            is_linear = (layer_index + 1) % interval != 0
            layer = SimpleNamespace(
                is_linear=is_linear,
                input_layernorm=SimpleNamespace(
                    weight=mx.ones((1,), dtype=mx.bfloat16)
                ),
            )
            if not is_linear:
                layer.self_attn = SimpleNamespace(
                    num_attention_heads=num_query_heads,
                    num_key_value_heads=num_kv_heads,
                    head_dim=head_dim,
                )
            layers.append(layer)
        return SimpleNamespace(model_type=model_type, args=args, layers=layers)

    def test_cache_layout_matches_both_target_architectures(self):
        for model_type in ("qwen3_5", "qwen3_5_moe"):
            with self.subTest(model_type=model_type):
                model = self.make_model(model_type)
                manager = QwenHybridPagedKVCacheManager(
                    model,
                    capacity_pages=8,
                    page_size=4,
                    dtype=mx.float32,
                )
                cache = manager.make_cache(sequence_id="request")
                self.assertEqual(len(cache), 4)
                self.assertIsInstance(cache[0], ArraysCache)
                self.assertIsInstance(cache[1], PagedKVCache)
                self.assertIsInstance(cache[2], ArraysCache)
                self.assertIsInstance(cache[3], PagedKVCache)
                self.assertEqual(manager.num_full_attention_layers, 2)
                manager.release(cache)

    def test_materialize_evaluates_physical_page_storage(self):
        manager = QwenHybridPagedKVCacheManager(
            self.make_model(),
            capacity_pages=2,
            page_size=4,
            dtype=mx.float32,
        )

        with patch("mlx_lm.models.paged_cache.mx.eval") as evaluate:
            manager.materialize()

        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(len(evaluate.call_args.args), 4)

    def test_strict_contract_matches_official_target_geometries(self):
        expected_full_layers = {"qwen3_5_moe": 10, "qwen3_5": 16}
        for model_type in ("qwen3_5_moe", "qwen3_5"):
            with self.subTest(model_type=model_type):
                model = self.make_target_stub(model_type)
                manager = QwenHybridPagedKVCacheManager(
                    model,
                    capacity_pages=1,
                    page_size=4,
                    strict_architecture=True,
                )
                self.assertEqual(
                    manager.num_full_attention_layers,
                    expected_full_layers[model_type],
                )
                self.assertTrue(
                    all(
                        pool.key_head_dim == 256
                        for pool in manager._pools
                        if pool is not None
                    )
                )

    def test_reference_path_matches_dense_qwen_forward(self):
        model = self.make_model()
        dense_cache = model.make_cache()
        manager = QwenHybridPagedKVCacheManager(
            model,
            capacity_pages=8,
            page_size=4,
            dtype=mx.float32,
        )
        paged_cache = manager.make_cache(sequence_id="paged")

        prompt = mx.array([[1, 2, 3, 4]], dtype=mx.uint32)
        dense_logits = model(prompt, cache=dense_cache)
        paged_logits = model(prompt, cache=paged_cache)
        mx.eval(dense_logits, paged_logits)
        self.assertTrue(mx.allclose(dense_logits, paged_logits, atol=1e-5, rtol=1e-5))

        token = mx.array([[5]], dtype=mx.uint32)
        dense_logits = model(token, cache=dense_cache)
        paged_logits = model(token, cache=paged_cache)
        mx.eval(dense_logits, paged_logits)
        self.assertTrue(mx.allclose(dense_logits, paged_logits, atol=1e-5, rtol=1e-5))

        stats = manager.stats()
        self.assertEqual(stats.full_attention_layers, 2)
        self.assertEqual(stats.live_pages, 4)
        manager.release(paged_cache)
        self.assertEqual(manager.stats().live_pages, 0)

    def test_admission_reserves_each_full_attention_pool_atomically(self):
        model = self.make_model()
        manager = QwenHybridPagedKVCacheManager(
            model,
            capacity_pages=2,
            page_size=4,
            dtype=mx.float32,
        )
        first = manager.make_cache(sequence_id="first")
        manager.admit("first", prompt_tokens=5, max_tokens=1, cache=first)
        stats = manager.stats()
        self.assertEqual(stats.live_pages, 4)
        self.assertEqual(stats.reserved_pages, 4)
        self.assertEqual(stats.free_pages, 0)

        second = manager.make_cache(sequence_id="second")
        with self.assertRaises(PageAllocationError):
            manager.admit("second", prompt_tokens=1, max_tokens=1, cache=second)
        self.assertEqual(manager.stats(), stats)

        self.assertTrue(manager.release_admission("first"))
        self.assertFalse(manager.release_admission("first"))
        self.assertEqual(manager.stats().live_pages, 0)
        manager.release(first)
        manager.release(second)

    def test_admission_covers_prefill_to_decode_tail_copy_on_write(self):
        model = self.make_model()
        manager = QwenHybridPagedKVCacheManager(
            model,
            capacity_pages=3,
            page_size=4,
            dtype=mx.float32,
        )
        generator = BatchGenerator(
            model,
            max_tokens=1,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=4,
        )

        initial = manager.make_cache()
        manager.admit(0, 4, 1, initial)
        (uid,) = generator.insert([[1, 2, 3, 4]], caches=[initial])
        self.assertEqual(uid, 0)
        self.assertEqual(manager.stats().reserved_pages, 6)

        (response,) = generator.next_generated()
        self.assertEqual(response.finish_reason, "length")
        manager.release_admission(uid)
        self.assertEqual(manager.stats().reserved_pages, 0)
        manager.release(initial)
        manager.release(response.prompt_cache)
        generator.close()
        self.assertEqual(manager.stats().live_pages, 0)

    def test_segment_admission_covers_retained_prefix_tail_copy_on_write(self):
        model = self.make_model()
        manager = QwenHybridPagedKVCacheManager(
            model,
            capacity_pages=4,
            page_size=4,
            dtype=mx.float32,
        )
        generator = BatchGenerator(
            model,
            max_tokens=1,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=4,
        )
        initial = manager.make_cache()
        manager.admit_segments(0, (2, 1, 1), 1, initial)
        (uid,) = generator.insert_segments(
            segments=[[[1, 2], [3], [4]]],
            caches=[initial],
            max_tokens=[1],
        )
        retained = []
        final_response = None
        for _ in range(5):
            prompt_responses, generation_responses = generator.next()
            for response in prompt_responses:
                if response.end_of_segment and not response.end_of_prompt:
                    cache, _ = generator.extract_cache([uid])[uid]
                    retained.append(cache)
            if generation_responses:
                final_response = generation_responses[0]
                break

        self.assertIsNotNone(final_response)
        self.assertEqual(final_response.finish_reason, "length")
        self.assertEqual(len(retained), 2)
        manager.release_admission(uid)
        manager.release(initial)
        self.assertEqual(manager.stats().reserved_pages, 0)
        for cache in retained:
            manager.release(cache)
        manager.release(final_response.prompt_cache)
        generator.close()
        self.assertEqual(manager.stats().live_pages, 0)

    def test_admission_rejects_request_larger_than_pool_capacity(self):
        model = self.make_model()
        manager = QwenHybridPagedKVCacheManager(
            model,
            capacity_pages=2,
            page_size=4,
            dtype=mx.float32,
        )
        cache = manager.make_cache(sequence_id="oversized")

        with self.assertRaisesRegex(
            RequestCapacityError, "requires 4 pages.*capacity is 2"
        ):
            manager.admit("oversized", prompt_tokens=8, max_tokens=1, cache=cache)

        stats = manager.stats()
        self.assertEqual(stats.live_pages, 0)
        self.assertEqual(stats.reserved_pages, 0)
        manager.release(cache)

    def test_cross_layer_admission_failure_restores_cache_metadata(self):
        model = self.make_model()
        manager = QwenHybridPagedKVCacheManager(
            model,
            capacity_pages=2,
            page_size=4,
            dtype=mx.float32,
        )
        full_pools = [pool for pool in manager._pools if pool is not None]
        blocker = full_pools[1].allocator.allocate(2)
        cache = manager.make_cache(sequence_id="original")
        before = [
            (layer.reservation_id, layer.sequence_id)
            for layer in cache
            if isinstance(layer, PagedKVCache)
        ]

        with self.assertRaises(PageAllocationError):
            manager.admit("request", prompt_tokens=1, max_tokens=1, cache=cache)

        after = [
            (layer.reservation_id, layer.sequence_id)
            for layer in cache
            if isinstance(layer, PagedKVCache)
        ]
        self.assertEqual(after, before)
        self.assertEqual(manager.stats().reserved_pages, 0)
        self.assertEqual(full_pools[0].stats().live_pages, 0)
        full_pools[1].allocator.release(blocker)
        manager.release(cache)

    def test_rejects_models_outside_target_scope(self):
        model = self.make_model()
        model.model_type = "llama"
        with self.assertRaises(ValueError):
            QwenHybridPagedKVCacheManager(model, capacity_pages=8)


class TestBatchCapacityBoundary(unittest.TestCase):
    def test_exhaustion_does_not_commit_an_earlier_row(self):
        pool = KVBlockPool(
            capacity_pages=1,
            page_size=4,
            num_kv_heads=1,
            key_head_dim=8,
            dtype=mx.float32,
        )
        batch = BatchPagedKVCache([PagedKVCache(pool), PagedKVCache(pool)])
        batch.prepare(lengths=[1, 1])
        x = mx.ones((2, 1, 1, 8))
        with self.assertRaises(PageAllocationError):
            batch.update_and_fetch(x, x)
        self.assertEqual([c.offset for c in batch.caches], [0, 0])
        self.assertEqual(batch._remaining_lengths, [1, 1])
        self.assertEqual(pool.stats().live_pages, 0)
        batch.release()

    def test_invalid_value_heads_do_not_mutate_storage(self):
        pool = KVBlockPool(
            capacity_pages=2,
            page_size=4,
            num_kv_heads=1,
            key_head_dim=8,
            dtype=mx.float32,
        )
        batch = BatchPagedKVCache([PagedKVCache(pool)])
        with self.assertRaises(ValueError):
            batch.update_and_fetch(mx.ones((1, 1, 1, 8)), mx.ones((1, 2, 1, 8)))
        self.assertEqual(pool.stats().live_pages, 0)
        batch.release()
