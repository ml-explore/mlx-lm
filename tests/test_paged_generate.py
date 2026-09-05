import gc
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.generate import BatchGenerator
from mlx_lm.models.paged_cache import PageAllocationError, RequestCapacityError
from mlx_lm.paged_generate import PagedBatchGenerator
import test_paged_cache


class TestPagedGenerator(unittest.TestCase):
    def setUp(self):
        self.model = test_paged_cache.TestQwenHybridPagedKVCacheManager().make_model()
        self.generator = PagedBatchGenerator(
            self.model,
            capacity_pages=32,
            page_size=4,
            max_tokens=4,
            prefill_step_size=4,
        )

    def tearDown(self):
        self.generator.close()

    def assert_released(self):
        gc.collect()
        stats = self.generator.cache_manager.stats()
        self.assertEqual(stats.reserved_pages, 0)
        self.assertEqual(stats.live_pages, 0)

    def test_output_matches_contiguous_and_completion_releases(self):
        prompts = [[1, 2, 3, 4, 5], [7, 8]]

        def collect(generator):
            output = {uid: [] for uid in generator.insert(prompts)}
            while responses := generator.next_generated():
                for response in responses:
                    output[response.uid].append(response.token)
            return list(output.values())

        dense = BatchGenerator(self.model, max_tokens=4, prefill_step_size=4)
        try:
            self.assertEqual(collect(dense), collect(self.generator))
        finally:
            dense.close()
        self.assert_released()

    def test_cancel_queued_prefill_and_decode(self):
        for steps in (0, 1, 4):
            with self.subTest(steps=steps):
                uids = self.generator.insert([[1, 2, 3, 4, 5]], max_tokens=[20])
                for _ in range(steps):
                    self.generator.next()
                self.generator.remove(uids)
                self.assert_released()

    def test_failed_multi_admission_is_atomic(self):
        for limits in ([4, 1000], [4, 116]):
            with self.subTest(limits=limits):
                with self.assertRaises((PageAllocationError, RequestCapacityError)):
                    self.generator.insert([[1, 2], [3, 4]], max_tokens=limits)
                self.assertEqual(len(self.generator._unprocessed_sequences), 0)
                self.assert_released()

    def test_scheduler_validation_rolls_back_admission(self):
        with self.assertRaises(ValueError):
            self.generator.insert([[1, 2]], samplers=[lambda x: x, lambda x: x])
        self.assert_released()

    def test_model_failure_releases_and_generator_can_resume(self):
        self.generator.insert([[1, 2, 3, 4, 5]])
        with patch.object(
            type(self.model), "__call__", side_effect=RuntimeError("test failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.generator.next()
        self.assert_released()
        self.generator.insert([[1, 2]], max_tokens=[1])
        self.assertTrue(self.generator.next_generated())

    def test_rotating_cache_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rotating max_kv_size"):
            PagedBatchGenerator(self.model, capacity_pages=8, max_kv_size=128)

    def test_prefix_reuse_and_close(self):
        (uid,) = self.generator.insert([[1, 2, 3, 4, 5]], max_tokens=[1])
        (response,) = self.generator.next_generated()
        prefix = response.prompt_cache
        old_offsets = [c.offset for c in prefix if hasattr(c, "pool")]
        self.generator.insert([[response.token, 6]], caches=[prefix], max_tokens=[2])
        self.generator.next_generated()
        self.assertEqual(old_offsets, [c.offset for c in prefix if hasattr(c, "pool")])
        self.generator.close()
        self.generator.cache_manager.release(prefix)
        self.assert_released()
