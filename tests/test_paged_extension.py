import unittest
from dataclasses import replace
from unittest.mock import patch

import mlx.core as mx
import test_paged_generate
from eco_paged_attention import paged_attention

from mlx_lm.models.paged_cache import BatchPagedKVCache
from mlx_lm.models.qwen3_5 import TextModel
from mlx_lm.paged_generate import PagedBatchGenerator


class TestPagedExtension(unittest.TestCase):
    def test_invalid_metadata(self):
        q = mx.ones((1, 6, 1, 256))
        k = mx.ones((1, 1, 64, 256))
        for page, length in ((0, 0), (0, 65), (1, 1)):
            with self.subTest(page=page, length=length):
                with self.assertRaises(ValueError):
                    paged_attention(
                        q,
                        k,
                        k,
                        mx.array([[page]], dtype=mx.uint32),
                        mx.array([length], dtype=mx.uint32),
                        scale=1 / 16,
                    )

    def test_nondefault_stream_and_lazy_array_lifetime(self):
        stream = mx.new_stream(mx.gpu)
        with mx.stream(stream):
            q = mx.ones((1, 6, 1, 256))
            k = mx.ones((1, 1, 64, 256))
            out = paged_attention(
                q,
                k,
                k,
                mx.array([[0]], dtype=mx.uint32),
                mx.array([63], dtype=mx.uint32),
                scale=1 / 16,
            )
        del q, k
        mx.eval(out)
        self.assertTrue(mx.allclose(out, mx.ones_like(out)))

    def test_decode_matches_sdpa(self):
        mx.random.seed(19)
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for gqa in (6, 8):
                with self.subTest(dtype=dtype, gqa=gqa):
                    k = mx.random.normal((8, 2, 64, 256)).astype(dtype)
                    v = mx.random.normal(k.shape).astype(dtype)
                    q = mx.random.normal((2, 2 * gqa, 1, 256)).astype(dtype)
                    tables = mx.array([[7, 1, 3, 5], [0, 6, 2, 4]], dtype=mx.uint32)
                    lengths = mx.array([129, 251], dtype=mx.uint32)
                    out = paged_attention(q, k, v, tables, lengths, scale=1 / 16)
                    refs = []
                    for row, length in enumerate((129, 251)):
                        kr = (
                            k[tables[row]]
                            .transpose(1, 0, 2, 3)
                            .reshape(1, 2, -1, 256)[:, :, :length]
                        )
                        vr = (
                            v[tables[row]]
                            .transpose(1, 0, 2, 3)
                            .reshape(1, 2, -1, 256)[:, :, :length]
                        )
                        refs.append(
                            mx.fast.scaled_dot_product_attention(
                                q[row : row + 1].astype(mx.float32),
                                kr.astype(mx.float32),
                                vr.astype(mx.float32),
                                scale=1 / 16,
                            )
                        )
                    ref = mx.concatenate(refs)
                    mx.eval(out, ref)
                    tolerance = 0.004 if dtype == mx.bfloat16 else 0.0005
                    self.assertLess(
                        float(mx.max(mx.abs(out.astype(mx.float32) - ref))), tolerance
                    )


class TestPagedAttentionGenerator(test_paged_generate.TestPagedGenerator):
    def setUp(self):
        super().setUp()
        self.generator.close()
        self.model = TextModel(
            replace(
                self.model.args,
                head_dim=256,
                num_attention_heads=6,
                num_key_value_heads=1,
            )
        )
        self.generator = PagedBatchGenerator(
            self.model,
            capacity_pages=32,
            page_size=4,
            paged_attention=True,
            max_tokens=4,
            prefill_step_size=4,
        )

    def test_decode_does_not_gather(self):
        self.generator.insert([[1, 2, 3], [4, 5]], max_tokens=[4, 4])
        self.generator.next_generated()
        with patch.object(
            BatchPagedKVCache,
            "gather",
            side_effect=AssertionError("gather during decode"),
        ):
            self.assertTrue(self.generator.next_generated())

    def test_custom_mask_uses_sdpa(self):
        self.generator.insert([[1, 2, 3]], max_tokens=[4])
        self.generator.next_generated()
        cache = self.generator._generation_batch.prompt_cache[1]
        q = mx.ones((1, 6, 1, 256))
        k = mx.ones((1, 1, 1, 256))
        mask = mx.zeros((1, 1, 1, cache.size() + 1), dtype=mx.bool_)
        with patch.object(
            cache.pool,
            "attention",
            side_effect=AssertionError("custom mask dispatched"),
        ):
            out = cache.update_and_attend(q, k, k, scale=1 / 16, mask=mask)
            mx.eval(out)
