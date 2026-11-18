"""Tests for context sharding.

Run with two or more processes:

    mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- \
        python tests/sharded_context_tests.py

With one process the tests still run, but nothing is split between machines.
The file name does not match `test*.py`, so `unittest discover` skips it.
"""

import importlib
import os
import random
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.models import gemma3_text, gemma4_text
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache
from mlx_lm.models.sharded_cache import ShardedKVCache
from mlx_lm.sharded_prompt_cache import (
    _backbone,
    block_owners,
    block_ranges,
    load_sharded_cache,
    make_sharded_cache,
    prefill_in_blocks,
    query_scope,
    save_sharded_cache,
)

GROUP = mx.distributed.init()
BLOCK = 32
TOL = 2e-3

COMMON = dict(
    hidden_size=256,
    num_hidden_layers=4,
    intermediate_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-6,
    vocab_size=256,
)
G4 = {
    **COMMON,
    "model_type": "gemma4_text",
    "head_dim": 64,
    "global_head_dim": 64,
    "sliding_window": 24,
    "vocab_size_per_layer_input": 256,
}
FAMILIES = {
    "llama": dict(model_type="llama", head_dim=64, **COMMON),
    "qwen2": dict(model_type="qwen2", **COMMON),
    "qwen3": dict(
        model_type="qwen3",
        head_dim=64,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        **COMMON,
    ),
    "gemma": dict(model_type="gemma", head_dim=64, **COMMON),
    "gemma2": dict(
        model_type="gemma2", head_dim=64, query_pre_attn_scalar=64.0, **COMMON
    ),
    "gemma3_text": dict(
        model_type="gemma3_text",
        head_dim=64,
        query_pre_attn_scalar=64,
        sliding_window=16,
        sliding_window_pattern=2,
        **COMMON,
    ),
    # No shared layers and no per-layer inputs.
    "gemma4_plain": {
        **G4,
        "num_hidden_layers": 6,
        "sliding_window_pattern": 3,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
    },
    # The last 4 layers reuse the keys and values of earlier layers.
    "gemma4_shared": {
        **G4,
        "num_hidden_layers": 8,
        "sliding_window_pattern": 4,
        "num_kv_shared_layers": 4,
        "hidden_size_per_layer_input": 32,
    },
    # Global layers: keys equal values, own head size and own number of heads.
    "gemma4_keq": {
        **G4,
        "num_hidden_layers": 6,
        "sliding_window_pattern": 3,
        "global_head_dim": 128,
        "attention_k_eq_v": True,
        "num_global_key_value_heads": 1,
        "num_kv_shared_layers": 2,
        "hidden_size_per_layer_input": 0,
    },
}
# Stock mlx-lm cannot run shared layers on a quantized cache.
QUANTIZED_REFERENCE = {"qwen2", "gemma3_text", "gemma4_plain"}


def build(name):
    module = importlib.import_module(f"mlx_lm.models.{FAMILIES[name]['model_type']}")
    mx.random.seed(0)
    model = module.Model(module.ModelArgs(**FAMILIES[name]))
    mx.eval(model.parameters())
    return model


def tokens(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(1, 256) for _ in range(n)]


def plain_caches(model, kv_bits):
    caches = make_prompt_cache(model)
    if kv_bits:
        caches = [
            QuantizedKVCache(group_size=64, bits=kv_bits) if type(c) is KVCache else c
            for c in caches
        ]
    return caches


def max_diff(xs, ys):
    return max(mx.max(mx.abs(a - b)).item() for a, b in zip(xs, ys))


def same_top_token(xs, ys):
    return all(
        mx.argmax(a, axis=-1).item() == mx.argmax(b, axis=-1).item()
        for a, b in zip(xs, ys)
    )


def one_machine(model, ids, question, forced, kv_bits, emb=None, groups=None):
    """Logits after the question and after each forced token, with plain caches."""
    caches = plain_caches(model, kv_bits)
    backbone = _backbone(model)
    for a, b in block_ranges(len(ids), BLOCK, groups):
        kwargs = {}
        if emb is not None:
            kwargs = dict(
                input_embeddings=emb[:, a:b], image_groups=mx.array(groups[a:b])
            )
        mx.eval(backbone(mx.array(ids[a:b])[None], caches, **kwargs))
    logits = model(mx.array(question)[None], cache=caches)[:, -1, :]
    mx.eval(logits)
    outs = [logits]
    for t in forced:
        logits = model(mx.array([[t]], dtype=mx.int32), cache=caches)[:, -1, :]
        mx.eval(logits)
        outs.append(logits)
    return outs


def greedy_tokens(model, ids, question, steps, emb=None, groups=None):
    outs = one_machine(model, ids, question, [], None, emb, groups)
    forced = [mx.argmax(outs[0], axis=-1).item()]
    for _ in range(steps - 1):
        outs = one_machine(model, ids, question, forced, None, emb, groups)
        forced.append(mx.argmax(outs[-1], axis=-1).item())
    return forced


def sharded_prefill(model, ids, kv_bits, emb=None, groups=None):
    caches = make_sharded_cache(model, GROUP, kv_bits=kv_bits)
    n_blocks = len(block_ranges(len(ids), BLOCK, groups))
    owners = block_owners(n_blocks, GROUP.size())
    prefill_in_blocks(
        model, ids, caches, GROUP, BLOCK, owners, embeddings=emb, image_groups=groups
    )
    return caches


def sharded_answer(model, caches, base, question, forced):
    outs = []
    with query_scope(caches, base) as set_position:
        # Evaluate every pass before the next one, as generation does.
        logits = model(mx.array(question)[None], cache=caches)[:, -1, :]
        mx.eval(logits)
        outs.append(logits)
        for i, t in enumerate(forced):
            set_position(base + len(question) + i)
            logits = model(mx.array([[t]], dtype=mx.int32), cache=caches)[:, -1, :]
            mx.eval(logits)
            outs.append(logits)
    return outs


class ShardedFamiliesTest(unittest.TestCase):
    def check_family(self, name, kv_bits=None):
        model = build(name)
        ids, q1, q2 = tokens(1, 100), tokens(2, 6), tokens(3, 5)
        caches = sharded_prefill(model, ids, kv_bits)
        kinds = "".join("S" if isinstance(c, ShardedKVCache) else "l" for c in caches)
        self.assertIn("S", kinds)

        results = []
        for question in (q1, q2):  # the second question runs after a rollback
            forced = greedy_tokens(model, ids, question, 4)
            ref = one_machine(model, ids, question, forced, kv_bits)
            got = sharded_answer(model, caches, len(ids), question, forced)
            results.append((question, forced, ref, got))

        # Save the cache, load it again and ask the first question once more.
        prefix = os.path.join(tempfile.gettempdir(), f"sharded_context_{name}_{kv_bits}")
        path = save_sharded_cache(prefix, caches, GROUP, len(ids))
        loaded, _ = load_sharded_cache(prefix, GROUP)
        question, forced, ref, _ = results[0]
        results.append((question, forced, ref, sharded_answer(
            model, loaded, len(ids), question, forced)))
        for p in (path, prefix + f".rank{GROUP.rank()}of{GROUP.size()}.local.safetensors"):
            if os.path.exists(p):
                os.remove(p)

        for _, _, ref, got in results:
            self.assertLess(max_diff(got, ref), TOL)
            self.assertTrue(same_top_token(got, ref))

    def test_families(self):
        for name in FAMILIES:
            with self.subTest(name):
                self.check_family(name)

    def test_quantized_cache(self):
        for name in sorted(QUANTIZED_REFERENCE):
            with self.subTest(name):
                self.check_family(name, kv_bits=8)


class ShardedImagesTest(unittest.TestCase):
    """Two "images" (random vectors) inside a text. Tokens of one image attend
    to each other in both directions, and blocks never cut an image."""

    IMAGES = [(20, 36), (70, 82)]

    def build_gemma(self, family):
        common = dict(
            hidden_size=256,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rms_norm_eps=1e-6,
            vocab_size=256,
            sliding_window=24,
        )
        mx.random.seed(0)
        if family == "gemma3_text":
            args = gemma3_text.ModelArgs(
                model_type="gemma3_text",
                num_hidden_layers=4,
                query_pre_attn_scalar=64,
                sliding_window_pattern=2,
                **common,
            )
            model = gemma3_text.Model(args)
        else:
            args = gemma4_text.ModelArgs(
                model_type="gemma4_text",
                num_hidden_layers=8,
                global_head_dim=64,
                sliding_window_pattern=2,
                num_kv_shared_layers=2,
                hidden_size_per_layer_input=32,
                vocab_size_per_layer_input=256,
                **common,
            )
            model = gemma4_text.Model(args)
        mx.eval(model.parameters())
        return model

    def inputs(self, model):
        ids = tokens(1, 120)
        groups = [-1] * len(ids)
        for k, (a, b) in enumerate(self.IMAGES):
            for i in range(a, b):
                groups[i] = k
                ids[i] = 0  # an image position has no token identity
        emb = model.model.embed_tokens(mx.array(ids)[None])
        mx.random.seed(5)
        scale = mx.std(emb).item()
        for a, b in self.IMAGES:
            features = mx.random.normal((1, b - a, emb.shape[-1])) * scale
            emb = mx.concatenate([emb[:, :a], features, emb[:, b:]], axis=1)
        mx.eval(emb)
        return ids, groups, emb

    def exact(self, model, ids, groups, emb, question, forced):
        """No cache: run the whole sequence in one pass for every step."""
        seq_ids, seq_groups = ids + question, groups + [-1] * len(question)
        seq_emb = mx.concatenate(
            [emb, model.model.embed_tokens(mx.array(question)[None])], axis=1
        )
        outs = []
        for n in range(len(forced) + 1):
            if n:
                t = forced[n - 1]
                seq_ids, seq_groups = seq_ids + [t], seq_groups + [-1]
                new = model.model.embed_tokens(mx.array([[t]]))
                seq_emb = mx.concatenate([seq_emb, new], axis=1)
            # The model scales its input embeddings in place. Pass a copy.
            logits = model(
                mx.array(seq_ids)[None],
                input_embeddings=seq_emb * 1,
                image_groups=mx.array(seq_groups),
            )[:, -1, :]
            mx.eval(logits)
            outs.append(logits)
        return outs

    def check_family(self, family):
        model = self.build_gemma(family)
        ids, groups, emb = self.inputs(model)
        question = tokens(2, 6)
        ranges = block_ranges(len(ids), BLOCK, groups)
        for a, b in self.IMAGES:  # no block boundary inside an image
            self.assertFalse(any(a < end < b for _, end in ranges))

        forced = greedy_tokens(model, ids, question, 3, emb, groups)
        exact = self.exact(model, ids, groups, emb, question, forced)
        for bits in (None, 8, 4):
            caches = sharded_prefill(model, ids, bits, emb, groups)
            got = sharded_answer(model, caches, len(ids), question, forced)
            try:
                ref = one_machine(model, ids, question, forced, bits, emb, groups)
            except Exception:  # stock mlx-lm cannot run shared layers on this cache
                ref = None
            if ref is not None:
                self.assertLess(max_diff(got, ref), TOL)
            if bits is None:
                self.assertLess(max_diff(got, exact), TOL)
            else:
                self.assertTrue(max_diff(got, exact) < 5.0)  # finite and near

    def test_gemma3_images(self):
        self.check_family("gemma3_text")

    def test_gemma4_images(self):
        self.check_family("gemma4_text")


if __name__ == "__main__":
    unittest.main()
