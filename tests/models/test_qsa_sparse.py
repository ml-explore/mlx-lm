# Copyright © 2026 Apple Inc.

import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from itertools import product
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.kernels import sparse_attention
from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import (
    CacheList,
    KVCache,
    QuantizedKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.models.qwen4_exp import layers

TEXT_CONFIG = dict(
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,  # The Metal kernel does not run: tests cover the gather path.
    vocab_size=256,
    rms_norm_eps=1e-6,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    hc_count=2,
    hc_lowrank=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=101,
    ple_embed_dim=64,
    ple_layer_ids=[2],
    eos_token_id=1,
    rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25},
)


def stock_mask(indexer, x, cache):
    """The dense mask of the stock indexer (one sequence, no padding)."""
    B, S, _ = x.shape
    r = indexer.compress_ratio
    qk = indexer.index_qk_proj(x)
    split = indexer.n_heads * indexer.head_dim
    q = qk[..., :split].reshape(B, S, indexer.n_heads, indexer.head_dim)
    raw_k = qk[..., split:].reshape(B, S, indexer.head_dim)
    offset = cache.offset
    raw_k = cache.update_and_fetch(raw_k[:, None], raw_k[:, None])[0][:, 0]
    kv_len = raw_k.shape[1]
    if kv_len <= indexer.token_budget:
        return None
    n_blocks = kv_len // r
    pooled = raw_k[:, : n_blocks * r].reshape(B, n_blocks, r, -1)
    pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
    pooled = indexer.block_rope(indexer.k_layernorm(pooled))
    q = indexer.rope(indexer.q_layernorm(q).transpose(0, 2, 1, 3), offset=offset)
    scores = q.astype(mx.float32) @ pooled[:, None].astype(mx.float32).swapaxes(-1, -2)
    scores = mx.maximum(scores, 0).sum(axis=1) / math.sqrt(indexer.head_dim)
    n_complete = (offset + mx.arange(S)[:, None] + 1) // r
    visible = mx.arange(n_blocks) < n_complete
    scores = mx.where(visible, scores, -mx.inf)
    k = min(indexer.block_topk, n_blocks)
    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    picked = mx.take_along_axis(visible[None], top, axis=-1)
    tok = mx.where(picked[..., None], top[..., None] * r + mx.arange(r), kv_len)
    keep = mx.zeros((B, S, kv_len + 1), dtype=mx.bool_)
    keep = mx.put_along_axis(keep, tok.reshape(B, S, -1), mx.array(True), axis=-1)
    cols = mx.arange(kv_len)
    q_cols = mx.arange(kv_len - S, kv_len)[:, None]
    own = (cols >= n_complete * r) & (cols <= q_cols)
    return (keep[..., :kv_len] | own)[:, None]


@contextmanager
def count_paths():
    """Count the attention calls past the budget: (picked keys only, dense mask)."""
    sparse = layers.Attention._sparse_attention
    with (
        patch.object(
            layers.Attention, "_sparse_attention", autospec=True, side_effect=sparse
        ) as s,
        patch.object(layers, "_key_mask", wraps=layers._key_mask) as m,
    ):
        yield lambda: (s.call_count, m.call_count)


def model_for(dtype, **kwargs):
    mx.random.seed(0)
    args = qwen4_exp.ModelArgs(
        model_type="qwen4_exp", text_config={**TEXT_CONFIG, **kwargs}
    )
    model = qwen4_exp.Model(args)
    model.set_dtype(dtype)
    model.eval()
    return model


class TestQSASparse(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()

    def tearDown(self):
        mx.set_default_device(self.device)

    def devices(self):
        return [mx.cpu, mx.gpu] if mx.metal.is_available() else [mx.cpu]

    def test_selection_matches_stock(self):
        # The indexer picks the keys of the stock indexer: chunks across block edges,
        # few key blocks per top-k step (1: fewer than k), trims to a block edge and
        # into a block, and a call without a cache. Its cache holds the stock block
        # keys (on the CPU, the float32 RMSNorm result changes with the row count).
        # Chunk sizes; "trim n" trims n tokens; "keys" checks the block keys.
        steps = [7, 1, 5, 3, 1, 1, 1, 22, 2, 1, 1, 17, "keys", "trim 6", 9, 1, 1, 1]
        steps += ["keys", "trim 7", 2, 1, 6, "keys"]
        for device, dtype, key_block in product(
            self.devices(), (mx.float32, mx.float16, mx.bfloat16), (4096, 3, 1)
        ):
            mx.set_default_device(device)
            indexer = model_for(dtype).layers[3].self_attn.indexer
            stock, cache = KVCache(), KVCache()
            for i, n in enumerate(steps):
                if isinstance(n, str) and n.startswith("trim"):
                    stock.trim(int(n[5:]))
                    cache.trim(int(n[5:]))
                    continue
                if n == "keys":
                    n_blocks = cache.offset // 4
                    raw = cache.keys[:, 0, : n_blocks * 4].reshape(1, n_blocks, 4, -1)
                    ref = raw.astype(mx.float32).mean(axis=2).astype(raw.dtype)
                    ref = indexer.block_rope(indexer.k_layernorm(ref))
                    out = cache.values[:, 0, 3 : n_blocks * 4 : 4]
                    if device == mx.gpu:
                        self.assertTrue(mx.array_equal(out, ref), (dtype, i))
                    elif dtype == mx.float32:
                        self.assertTrue(mx.allclose(out, ref, rtol=0, atol=1e-5), i)
                    continue
                x = mx.random.normal((1, n, 64), key=mx.random.key(i)).astype(dtype)
                ref = stock_mask(indexer, x, stock)
                with patch.object(layers, "_TOPK_KEY_BLOCK", key_block):
                    out = indexer(x, cache)
                if ref is None:
                    self.assertIsNone(out)
                    continue
                # The positions stay in the keys (the own block can run past them).
                self.assertTrue((out[0] < cache.offset).all().item())
                mask = layers._key_mask(*out, cache.offset)
                self.assertTrue(mx.array_equal(mask, ref), (device, dtype, i))
            # Without a cache.
            x = mx.random.normal((1, 22, 64), key=mx.random.key(99)).astype(dtype)
            out = indexer(x, None)
            self.assertTrue((out[0] < x.shape[1]).all().item())
            ref = stock_mask(indexer, x, KVCache())
            self.assertTrue(mx.array_equal(layers._key_mask(*out, x.shape[1]), ref))

    def test_sparse_matches_masked(self):
        # Attention over the picked keys only matches the masked dense attention: a
        # prompt shorter than a block, decode and chunks across block edges and the
        # budget, with small query and key blocks (the query blocks do not change the
        # result), then batch decode with left padding. On the CPU, the chunks also
        # match one call without a cache. (On the GPU, chunks and one call round
        # differently, which can change near ties of the top k.)
        chunks = [(0, 3), (3, 4), (4, 5), (5, 9), (9, 10), (10, 20), (20, 43)]
        chunks += [(43, 44), (44, 45), (45, 46)]
        for device in self.devices():
            mx.set_default_device(device)
            # The MoE gather matmul runs only in float32 on the CPU.
            dtypes = [(mx.float32, 1e-4)]
            if device == mx.gpu:
                dtypes += [(mx.float16, 2e-2), (mx.bfloat16, 5e-2)]
            for dtype, atol in dtypes:
                sparse = model_for(dtype, sparse_prefill_min_context=0)
                dense = model_for(dtype, sparse_attention=False)
                ids = mx.random.randint(2, 256, (2, 60), key=mx.random.key(1))
                caches = [make_prompt_cache(m) for m in (sparse, sparse, dense)]
                outs = []
                for s, e in chunks:
                    x = ids[:1, s:e]
                    with patch.object(layers, "_TOPK_KEY_BLOCK", 3):
                        with count_paths() as counts:
                            a = sparse(x, cache=caches[0])
                        with patch.object(layers, "_QUERY_BLOCK", 2):
                            a2 = sparse(x, cache=caches[1])
                    b = dense(x, cache=caches[2])
                    self.assertTrue(mx.array_equal(a, a2), (device, dtype, s))
                    self.assertTrue(mx.allclose(a, b, atol=atol), (device, dtype, s))
                    # One QSA layer: past the budget, the sparse path ran.
                    self.assertEqual(counts(), (int(e > 8), 0), (device, dtype, s))
                    outs.append(a)
                if device == mx.cpu:
                    ref = sparse(ids[:1, :46])
                    out = mx.concatenate(outs, axis=1)
                    self.assertTrue(mx.allclose(out, ref, atol=1e-4))
                # Batch decode with left padding: a row with state and a fresh row.
                merged = []
                for m, c in zip((sparse, dense), caches[::2]):
                    fresh = make_prompt_cache(m)
                    merged.append([x.merge([x, y]) for x, y in zip(c, fresh)])
                for i in range(46, 52):
                    x = ids[:, i : i + 1]
                    with count_paths() as counts:
                        a = sparse(x, cache=merged[0])
                    b = dense(x, cache=merged[1])
                    self.assertTrue(mx.allclose(a, b, atol=atol), (device, dtype, i))
                    # Batch decode also reads only the picked keys.
                    self.assertEqual(counts(), (1, 0), (device, dtype, i))

    @unittest.skipIf(not mx.metal.is_available(), "The kernel needs the Metal backend")
    def test_fused_matches_gather_path(self):
        # The Metal kernel (head_dim 64 is supported) in each variant, for prefill
        # chunks and a call without a cache, matches the gather and masked paths.
        # Without sparse_prefill_fused, the kernel is not called.
        mx.set_default_device(mx.gpu)
        kernel, calls = sparse_attention.indexed_attention, []

        def record(*args, **kwargs):
            out = kernel(*args, **kwargs)
            calls.append(out is not None)
            return out

        variants = [
            v for v in sparse_attention.VARIANTS if sparse_attention.is_available(v)
        ]
        dtypes = [(mx.float16, 2e-2), (mx.bfloat16, 5e-2)]
        chunks = [(0, 20), (20, 43), (43, 44), (44, 50)]
        for variant, (dtype, atol) in product(variants, dtypes):
            ids = mx.random.randint(2, 256, (1, 50), key=mx.random.key(1))
            outs = []
            for sparse, fused in [(True, True), (True, False), (False, False)]:
                model = model_for(
                    dtype,
                    head_dim=64,
                    sparse_attention=sparse,
                    sparse_prefill_min_context=0,
                    sparse_prefill_fused=fused,
                )
                cache = make_prompt_cache(model)
                with (
                    count_paths() as counts,
                    patch.object(sparse_attention, "indexed_attention", record),
                    patch.object(
                        sparse_attention, "default_variant", return_value=variant
                    ),
                ):
                    out = [model(ids[:, s:e], cache=cache) for s, e in chunks]
                    outs.append(out + [model(ids)])
                self.assertEqual(counts(), (5, 0) if sparse else (0, 5), variant)
            self.assertEqual(calls, [True] * 4, variant)
            calls.clear()
            for a, b, c in zip(*outs):
                self.assertTrue(mx.allclose(a, b, atol=atol), (variant, dtype))
                self.assertTrue(mx.allclose(a, c, atol=atol), (variant, dtype))

    def test_dispatch(self):
        # Prefill below sparse_prefill_min_context and batch prefill use the masked
        # path; decode always reads the picked keys. Training and a hand-built quantized
        # cache use the masked path, with the outputs of sparse_attention=False.
        args = qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=TEXT_CONFIG)
        self.assertTrue(args.text.sparse_attention)
        self.assertEqual(args.text.sparse_prefill_min_context, 32768)
        self.assertTrue(args.text.sparse_prefill_fused)
        for device in self.devices():
            mx.set_default_device(device)
            # min_context applies when the kernel does not run, also for sizes and a
            # dtype it supports (head_dim 64, bfloat16 on the GPU).
            model = model_for(
                mx.float32 if device == mx.cpu else mx.bfloat16,
                head_dim=64,
                sparse_prefill_min_context=30,
                sparse_prefill_fused=False,
            )
            ids = mx.random.randint(2, 256, (1, 40), key=mx.random.key(9))
            cache = make_prompt_cache(model)
            with patch.object(sparse_attention, "indexed_attention") as kernel:
                with count_paths() as counts:
                    model(ids[:, :20], cache=cache)
                self.assertEqual(counts(), (0, 1), device)
                with count_paths() as counts:
                    for s, e in [(20, 21), (21, 22), (22, 35)]:
                        model(ids[:, s:e], cache=cache)
                self.assertEqual(counts(), (3, 0), device)
            kernel.assert_not_called()
            cache = make_prompt_cache(model)
            with count_paths() as counts:
                model(mx.concatenate([ids[:, :30], ids[:, 10:40]]), cache=cache)
            self.assertEqual(counts(), (0, 1), device)
            with count_paths() as counts:
                for i in range(30, 40):
                    x = ids[:, i : i + 1]
                    model(mx.concatenate([x, x]), cache=cache)
            self.assertEqual(counts(), (10, 0), device)

            # Training past the budget: the loss and gradients of the stock model (the
            # selection does not get gradients). GPU gradients can differ in the last
            # bits between two runs.
            dtype = mx.float32 if device == mx.cpu else mx.bfloat16
            ids = mx.random.randint(2, 256, (1, 40), key=mx.random.key(5))
            losses, grads = [], []
            for sparse in (True, False):
                # head_dim 64: a size the Metal kernel supports.
                model = model_for(
                    dtype,
                    head_dim=64,
                    sparse_attention=sparse,
                    sparse_prefill_min_context=0,
                )
                model.train()

                def loss_fn(model):
                    logits = model(ids[:, :-1]).astype(mx.float32)
                    return nn.losses.cross_entropy(logits, ids[:, 1:]).mean()

                with count_paths() as counts:
                    loss, grad = nn.value_and_grad(model, loss_fn)(model)
                self.assertEqual(counts(), (0, 1), device)
                losses.append(loss)
                grads.append(grad)
            self.assertTrue(mx.array_equal(*losses), device)
            for (k, x), (_, y) in zip(*(tree_flatten(g) for g in grads)):
                close = mx.allclose(x, y, rtol=1e-2, atol=1e-4)
                self.assertTrue(close, (device, k))

            # A quantized attention cache, below and past the budget.
            ids = mx.random.randint(2, 256, (1, 30), key=mx.random.key(10))
            outs = []
            for sparse in (True, False):
                model = model_for(
                    mx.float32,
                    sparse_attention=sparse,
                    sparse_prefill_min_context=0,
                )
                cache = [
                    (
                        CacheList(QuantizedKVCache(group_size=32), c[1])
                        if isinstance(c, CacheList)
                        else c
                    )
                    for c in make_prompt_cache(model)
                ]
                with count_paths() as counts:
                    out = [model(ids[:, :6], cache=cache)]
                    out += [model(ids[:, i : i + 1], cache=cache) for i in range(6, 12)]
                    out.append(model(ids[:, 12:], cache=cache))
                # Past the budget: 4 decode steps and the last chunk.
                self.assertEqual(counts(), (0, 5), device)
                outs.append(mx.concatenate(out, axis=1))
            self.assertTrue(mx.array_equal(*outs), device)

    def test_padded_rows_match_alone(self):
        # A padded row of a batch matches the same row alone: left padding (a fresh
        # row merged next to one with state) and right padding (batch generator). On
        # the GPU, B = 2 and B = 1 round differently, which can change near ties of the
        # top k: there the batch is only compared with the masked path.
        for device in self.devices():
            mx.set_default_device(device)
            sparse = model_for(mx.float32, sparse_prefill_min_context=0)
            dense = model_for(mx.float32, sparse_attention=False)
            a = mx.random.randint(2, 256, (1, 50), key=mx.random.key(3))
            b = mx.random.randint(2, 256, (1, 30), key=mx.random.key(4))
            # Enough decode steps that blocks of the padded row end in them.
            steps = mx.random.randint(2, 256, (1, 16), key=mx.random.key(7))
            steps = mx.concatenate([steps, steps])

            def decode(model, cache, B):
                return [model(steps[:B, i : i + 1], cache=cache) for i in range(16)]

            def alone(n):
                cache = make_prompt_cache(sparse)
                out = [sparse(b[:, :n], cache=cache)[:, -1:]]
                return mx.concatenate(out + decode(sparse, cache, 1), axis=1)

            def left(model):
                cache = make_prompt_cache(model)
                model(a[:, :25], cache=cache)
                fresh = make_prompt_cache(model)
                cache = [x.merge([x, y]) for x, y in zip(cache, fresh)]
                out = model(mx.concatenate([a[:, 25:45], b[:, :20]]), cache=cache)
                return mx.concatenate([out[:, -1:]] + decode(model, cache, 2), axis=1)

            def right(model):
                empty = zip(make_prompt_cache(model), make_prompt_cache(model))
                cache = [x.merge([x, y]) for x, y in empty]
                for c in cache:
                    c.prepare(lengths=[30, 18], right_padding=[0, 12])
                model(mx.concatenate([a[:, :30], b[:, :30]]), cache=cache)
                for c in cache:
                    c.finalize()
                return mx.concatenate(decode(model, cache, 2), axis=1)

            for batch, n, first in [(left, 20, 0), (right, 18, 1)]:
                out = batch(sparse)
                self.assertTrue(mx.allclose(out, batch(dense), atol=1e-4), device)
                if device == mx.cpu:
                    ref = alone(n)[:, first:]
                    self.assertTrue(mx.allclose(out[1:], ref, atol=1e-4), n)

    def test_prompt_cache_save_load(self):
        model = model_for(mx.float32)
        ids = mx.random.randint(2, 256, (1, 40), key=mx.random.key(2))
        cache = make_prompt_cache(model)
        model(ids[:, :30], cache=cache)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "cache.safetensors")
            save_prompt_cache(f, cache)
            loaded = load_prompt_cache(f)
        for i in range(30, 35):
            a = model(ids[:, i : i + 1], cache=cache)
            b = model(ids[:, i : i + 1], cache=loaded)
            self.assertTrue(mx.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
