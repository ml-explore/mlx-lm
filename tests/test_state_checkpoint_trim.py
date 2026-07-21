# Copyright © 2026 Apple Inc.
"""Prefix-cache trim for linear-attention hybrid caches.

Hybrid models (Qwen3-Next / Kimi-Linear class) mix ArraysCache (recurrent
state, not trimmable backward) with KVCache. These tests cover the
prefill-time state checkpoints recorded on ArraysCache and the coordinated
partial trim that lands on a checkpoint boundary, including the
LRUPromptCache reuse path the server drives.
"""

import copy
import importlib
import os
import unittest

# Keep fp32 GEMMs off the TF32 path so the checkpoint-restore
# equivalence checks stay fp32-exact on M5-class devices.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx

from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    LRUPromptCache,
    RotatingKVCache,
    achievable_trim,
    can_trim_prompt_cache,
    make_prompt_cache,
    record_state_checkpoints,
    trim_prompt_cache,
)

QWEN3_NEXT_CONFIG = {
    "model_type": "qwen3_next",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": 1000,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 3,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "shared_expert_intermediate_size": 128,
    "mlp_only_layers": [0],
    "moe_intermediate_size": 128,
    "rms_norm_eps": 1e-5,
    "head_dim": 64,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 1000,
}

KIMI_LINEAR_CONFIG = {
    "model_type": "kimi_linear",
    "vocab_size": 1000,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 128,
    "head_dim": 32,
    "rope_theta": 100.0,
    "rms_norm_eps": 1e-6,
    "linear_attn_config": {
        "num_heads": 8,
        "head_dim": 32,
        "kda_layers": [1],
    },
    "model_max_length": 1000,
    "num_experts": 2,
    "moe_intermediate_size": 128,
    "kv_lora_rank": 8,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "v_head_dim": 16,
}


def make_model(config):
    arch = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    model = arch.Model(arch.ModelArgs.from_dict(config))
    model.eval()
    return model


def prefill(model, cache, tokens, chunk):
    """Chunked prefill mirroring generate_step: record a checkpoint at every
    chunk boundary and force one at the end. Returns the last chunk logits."""
    base = max((c.size() for c in cache), default=0)
    logits = None
    processed = 0
    for i in range(0, len(tokens), chunk):
        seg = mx.array(tokens[i : i + chunk])[None]
        logits = model(seg, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        processed += seg.shape[1]
        record_state_checkpoints(cache, [base + processed])
    if processed > 0:
        record_state_checkpoints(cache, [base + processed], force=True)
    return logits


def greedy(model, cache, last_logits, n):
    ids = []
    y = mx.argmax(last_logits[:, -1, :], axis=-1)
    for _ in range(n):
        ids.append(int(y.item()))
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
    return ids


def replay(model, cache, ids):
    """Feed known token ids one step at a time (decode path)."""
    logits = None
    for t in ids:
        logits = model(mx.array([[t]]), cache=cache)
    mx.eval(logits, [c.state for c in cache])
    return logits


class TestStateCheckpointTrim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env = {
            k: os.environ.get(k)
            for k in ("MLX_LM_STATE_CHECKPOINT_STRIDE", "MLX_LM_STATE_CHECKPOINT_MAX")
        }
        os.environ["MLX_LM_STATE_CHECKPOINT_STRIDE"] = "32"
        os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "8"

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---------------- synthetic caches ----------------

    def _synthetic_hybrid(self, boundaries):
        """KVCache + ArraysCache advanced through the given chunk boundaries."""
        kv = KVCache()
        ar = ArraysCache(size=2)
        cache = [kv, ar]
        states = {}
        pos = 0
        for b in boundaries:
            n = b - pos
            k = mx.random.normal((1, 2, n, 4))
            kv.update_and_fetch(k, k)
            ar[0] = mx.random.normal((1, 3))
            ar[1] = mx.random.normal((1, 5))
            pos = b
            record_state_checkpoints(cache, [pos])
            states[pos] = [mx.array(ar[0]), mx.array(ar[1])]
        record_state_checkpoints(cache, [pos], force=True)
        return cache, states

    def test_partial_trim_lands_on_checkpoint(self):
        cache, states = self._synthetic_hybrid([32, 64, 96, 113])
        kv, ar = cache
        self.assertFalse(can_trim_prompt_cache(cache))

        # Requested landing 95 is between checkpoints; snaps back to 64.
        self.assertEqual(achievable_trim(cache, 18), (64, 49))
        # Exact-checkpoint landing stays exact.
        self.assertEqual(achievable_trim(cache, 17), (96, 17))

        n = trim_prompt_cache(cache, 18, allow_partial=True)
        self.assertEqual(n, 49)
        self.assertEqual(kv.offset, 64)
        for got, want in zip([ar[0], ar[1]], states[64]):
            self.assertTrue(mx.allclose(got, want).item())
        # Checkpoints past the landing were dropped.
        self.assertEqual(ar.snap_trim_position(1000), 64)

    def test_partial_trim_is_opt_in(self):
        cache, _ = self._synthetic_hybrid([32, 64])
        kv, ar = cache
        state_before = [mx.array(ar[0]), mx.array(ar[1])]
        self.assertEqual(trim_prompt_cache(cache, 10), 0)
        self.assertEqual(kv.offset, 64)
        for got, want in zip([ar[0], ar[1]], state_before):
            self.assertTrue(mx.allclose(got, want).item())

    def test_trim_below_oldest_checkpoint_resets(self):
        cache, _ = self._synthetic_hybrid([32, 64])
        kv, ar = cache
        # target 54 -> lands on the checkpoint at 32
        n = trim_prompt_cache(cache, 10, allow_partial=True)
        self.assertEqual(n, 32)
        self.assertEqual(kv.offset, 32)
        # target 12 is below every checkpoint -> full reset
        n = trim_prompt_cache(cache, 20, allow_partial=True)
        self.assertEqual(n, 32)
        self.assertEqual(kv.offset, 0)
        self.assertTrue(ar.empty())

    def test_checkpoint_cap_and_thinning(self):
        os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "3"
        try:
            cache, _ = self._synthetic_hybrid([32, 64, 96, 128, 160, 192])
            ar = cache[1]
            lane = ar._checkpoints[0]
            self.assertLessEqual(len(lane), 3)
            # The newest (end-of-prefill) checkpoint always survives.
            self.assertEqual(lane[-1][0], 192)
        finally:
            os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "8"

    def test_disabled_via_env(self):
        os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "0"
        try:
            cache, _ = self._synthetic_hybrid([32, 64])
            ar = cache[1]
            self.assertEqual(ar._checkpoints, [])
            # Only the implicit empty state remains reachable.
            self.assertEqual(achievable_trim(cache, 10), (0, 64))
        finally:
            os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "8"

    def test_deepcopy_preserves_checkpoints(self):
        cache, states = self._synthetic_hybrid([32, 64, 96])
        clone = copy.deepcopy(cache)
        # target 56 -> lands on the checkpoint at 32
        n = trim_prompt_cache(clone, 40, allow_partial=True)
        self.assertEqual(n, 64)
        for got, want in zip([clone[1][0], clone[1][1]], states[32]):
            self.assertTrue(mx.allclose(got, want).item())
        # The original is untouched.
        self.assertEqual(cache[0].offset, 96)
        self.assertEqual(cache[1].snap_trim_position(1000), 96)

    # ---------------- rotating (sliding-window) caches ----------------

    def _rotating_hybrid(self, window, boundaries):
        """KVCache + RotatingKVCache advanced through the given boundaries,
        recording checkpoints. Returns the caches plus reference window
        copies (temporal order) captured at every boundary."""
        kv = KVCache()
        rot = RotatingKVCache(max_size=window)
        cache = [kv, rot]
        refs = {}
        pos = 0
        for b in boundaries:
            n = b - pos
            k = mx.random.normal((1, 2, n, 4))
            kv.update_and_fetch(k, k)
            rot.update_and_fetch(k, k)
            pos = b
            record_state_checkpoints(cache, [pos])
            refs[pos] = (
                mx.array(rot._temporal_order(rot.keys)),
                mx.array(rot._temporal_order(rot.values)),
            )
        record_state_checkpoints(cache, [pos], force=True)
        return cache, refs

    def test_rotating_hybrid_partial_trim(self):
        cache, refs = self._rotating_hybrid(32, [32, 64, 96, 113])
        kv, rot = cache
        # The ring has wrapped: not trimmable, so the strict path is closed.
        self.assertFalse(can_trim_prompt_cache(cache))
        # size() saturates at the window; the coordinator must use the
        # absolute position (113), not min(offset, max_size).
        self.assertEqual(achievable_trim(cache, 18), (64, 49))

        n = trim_prompt_cache(cache, 18, allow_partial=True)
        self.assertEqual(n, 49)
        self.assertEqual(kv.offset, 64)
        self.assertEqual(rot.offset, 64)
        self.assertTrue(mx.array_equal(rot.keys, refs[64][0]).item())
        self.assertTrue(mx.array_equal(rot.values, refs[64][1]).item())

        # The restored cache keeps working: advance again and re-trim.
        k = mx.random.normal((1, 2, 10, 4))
        kv.update_and_fetch(k, k)
        rot.update_and_fetch(k, k)
        n = trim_prompt_cache(cache, 42, allow_partial=True)
        self.assertEqual(n, 42)  # target 32 is an exact checkpoint
        self.assertEqual(rot.offset, 32)
        self.assertTrue(mx.array_equal(rot.keys, refs[32][0]).item())

    def test_rotating_wrapped_without_checkpoints_lands_at_zero(self):
        os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "0"
        try:
            cache, _ = self._rotating_hybrid(32, [64])
        finally:
            os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "8"
        kv, rot = cache
        # Only the empty state is reachable; the whole cache is trimmed.
        self.assertEqual(achievable_trim(cache, 10), (0, 64))
        n = trim_prompt_cache(cache, 10, allow_partial=True)
        self.assertEqual(n, 64)
        self.assertEqual(kv.offset, 0)
        self.assertIsNone(rot.keys)
        self.assertEqual(rot.offset, 0)

    def test_rotating_unwrapped_stays_natively_trimmable(self):
        cache, _ = self._rotating_hybrid(256, [32, 64])
        kv, rot = cache
        self.assertTrue(can_trim_prompt_cache(cache))
        self.assertEqual(trim_prompt_cache(cache, 10), 10)
        self.assertEqual(rot.offset, 54)

    # ---------------- batch lanes ----------------

    def test_batch_lanes_record_extract_extend(self):
        ar = ArraysCache(size=1)
        ar[0] = mx.random.normal((2, 3))
        record_state_checkpoints([ar], [40, 32], force=True)
        state0 = mx.array(ar[0])
        ar[0] = mx.random.normal((2, 3))
        record_state_checkpoints([ar], [80, 32], force=True)

        # Lane 1 was frozen at 32: only one (monotone) record kept.
        self.assertEqual([p for p, _ in ar._checkpoints[1]], [32])
        self.assertEqual([p for p, _ in ar._checkpoints[0]], [40, 80])

        # Batched caches never snap (extract slices them apart first).
        self.assertIsNone(ar.snap_trim_position(100))

        lane0 = ar.extract(0)
        self.assertEqual(lane0.snap_trim_position(50), 40)
        lane0.trim_to_position(40, 40)
        self.assertTrue(mx.allclose(lane0[0], state0[0:1]).item())

        # extend preserves both sides' histories.
        other = ArraysCache(size=1)
        other[0] = mx.random.normal((1, 3))
        record_state_checkpoints([other], [16], force=True)
        base = ar.extract(1)
        base.extend(other)
        self.assertEqual([p for p, _ in base._checkpoints[0]], [32])
        self.assertEqual([p for p, _ in base._checkpoints[1]], [16])

        # merge carries per-lane histories back into a batch cache.
        merged = ArraysCache.merge([lane0, other])
        self.assertEqual([p for p, _ in merged._checkpoints[0]], [40])
        self.assertEqual([p for p, _ in merged._checkpoints[1]], [16])

        # filter keeps the selected lanes' histories.
        merged.filter([1])
        self.assertEqual([p for p, _ in merged._checkpoints[0]], [16])

    # ---------------- LRU prompt cache path ----------------

    def _lru_roundtrip(self, config):
        """Serving scenario: stored entry = prompt + generated tail; new
        request repeats the prompt (regenerate). The fetch must land on a
        checkpoint and hand back the exact suffix to re-process."""
        mx.random.seed(0)
        model = make_model(config)
        chunk = 32
        prompt = mx.random.randint(0, config["vocab_size"], (96,)).tolist()
        tail = mx.random.randint(0, config["vocab_size"], (17,)).tolist()
        stored_key = prompt + tail

        stored_cache = make_prompt_cache(model)
        prefill(model, stored_cache, stored_key, chunk)
        self.assertFalse(can_trim_prompt_cache(stored_cache))

        lru = LRUPromptCache()
        lru.insert_cache("model-key", stored_key, stored_cache)

        cache, rest = lru.fetch_nearest_cache("model-key", prompt)
        self.assertIsNotNone(cache)
        # Landing must be a checkpoint at or before len(prompt) - 1 = 95,
        # and rest must be exactly the un-cached suffix of the prompt.
        landed = len(prompt) - len(rest)
        self.assertEqual(landed, 64)
        self.assertEqual(rest, prompt[64:])

        # Decode consistency: chunk-aligned landing makes the reused arm
        # bit-comparable with a fresh prefill of the same prompt.
        logits_reused = prefill(model, cache, rest, chunk)
        ids_reused = greedy(model, cache, logits_reused, 5)

        fresh_cache = make_prompt_cache(model)
        logits_fresh = prefill(model, fresh_cache, prompt, chunk)
        ids_fresh = greedy(model, fresh_cache, logits_fresh, 5)

        self.assertTrue(
            mx.allclose(
                logits_reused[:, -1, :], logits_fresh[:, -1, :], atol=1e-5
            ).item()
        )
        self.assertEqual(ids_reused, ids_fresh)

    def test_lru_fetch_qwen3_next(self):
        self._lru_roundtrip(QWEN3_NEXT_CONFIG)

    def test_lru_fetch_kimi_linear(self):
        self._lru_roundtrip(KIMI_LINEAR_CONFIG)

    def test_lru_fetch_prefers_deeper_exact_prefix(self):
        """If the longer entry's landing is shallower than an exact-prefix
        entry, the exact-prefix entry wins."""
        mx.random.seed(1)
        model = make_model(QWEN3_NEXT_CONFIG)
        chunk = 32
        prompt = mx.random.randint(0, 1000, (96,)).tolist()
        tail = mx.random.randint(0, 1000, (17,)).tolist()

        # Longer entry recorded WITHOUT checkpoints: it can only land at 0.
        os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "0"
        try:
            longer_cache = make_prompt_cache(model)
            prefill(model, longer_cache, prompt + tail, chunk)
        finally:
            os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] = "8"

        shorter_cache = make_prompt_cache(model)
        prefill(model, shorter_cache, prompt[:80], chunk)

        lru = LRUPromptCache()
        lru.insert_cache("model-key", prompt + tail, longer_cache)
        lru.insert_cache("model-key", prompt[:80], shorter_cache)

        cache, rest = lru.fetch_nearest_cache("model-key", prompt)
        self.assertIsNotNone(cache)
        self.assertEqual(rest, prompt[80:])


    # ------- state-level audit + through-decode round trip -------

    def _assert_state_trees_equal(self, a, b):
        if isinstance(a, mx.array):
            self.assertTrue(mx.array_equal(a, b).item())
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self._assert_state_trees_equal(x, y)
        else:
            self.assertEqual(a, b)

    def _checkpoint_state_audit(self, config):
        """Every recorded checkpoint must restore to a state bit-identical
        to a fresh prefill of the same prefix. Output-level checks alone
        can hide a wrong-state restore (downstream generations may look
        plausible and even benchmark-clean), so compare the cache state
        itself at every landing."""
        mx.random.seed(3)
        model = make_model(config)
        chunk = 32
        tokens = mx.random.randint(0, config["vocab_size"], (113,)).tolist()
        cache = make_prompt_cache(model)
        prefill(model, cache, tokens, chunk)
        size = max(c.size() for c in cache)
        self.assertEqual(size, 113)

        for p in (32, 64, 96):
            landing = achievable_trim(cache, size - p)
            self.assertIsNotNone(landing)
            self.assertEqual(landing[0], p)

            trimmed = copy.deepcopy(cache)
            actual = trim_prompt_cache(trimmed, size - p, allow_partial=True)
            self.assertEqual(actual, size - p)

            fresh = make_prompt_cache(model)
            prefill(model, fresh, tokens[:p], chunk)
            for ct, cf in zip(trimmed, fresh):
                self.assertEqual(ct.size(), cf.size())
                self._assert_state_trees_equal(ct.state, cf.state)

    def test_checkpoint_state_audit_qwen3_next(self):
        self._checkpoint_state_audit(QWEN3_NEXT_CONFIG)

    def test_checkpoint_state_audit_kimi_linear(self):
        self._checkpoint_state_audit(KIMI_LINEAR_CONFIG)

    def _generation_roundtrip(self, config):
        """Serving round trip THROUGH decode: prefill -> greedy decode ->
        store (forced end checkpoint) -> follow-up request reuses the
        generated prefix -> re-extend -> continue. Guards the
        wrong-position class of bugs where state committed during decode
        displaces or contaminates the checkpoint a later request restores.
        The reference arm replicates the identical computation path
        (chunked prefill + per-token replay) with no store/fetch/trim, so
        the reuse machinery must be numerically a no-op: the continuation
        is required to be token-exact."""
        mx.random.seed(2)
        model = make_model(config)
        chunk = 32
        vocab = config["vocab_size"]
        prompt = mx.random.randint(0, vocab, (96,)).tolist()
        extra = mx.random.randint(0, vocab, (7,)).tolist()

        # Serving pass: prefill, decode 24 tokens, store prompt+generated.
        cache = make_prompt_cache(model)
        logits = prefill(model, cache, prompt, chunk)
        generated = greedy(model, cache, logits, 24)
        record_state_checkpoints(
            cache, [max(c.size() for c in cache)], force=True
        )
        stored_key = prompt + generated
        lru = LRUPromptCache()
        lru.insert_cache("model-key", stored_key, cache)

        # Follow-up: the conversation continues past the generated tail.
        # The landing must be the forced end-of-generation checkpoint at
        # 120 — i.e. the decode-produced state is what gets reused.
        request = stored_key + extra
        reused, rest = lru.fetch_nearest_cache("model-key", request)
        self.assertIsNotNone(reused)
        self.assertEqual(len(request) - len(rest), 120)
        self.assertEqual(rest, extra)

        logits_reused = prefill(model, reused, rest, chunk)
        cont_reused = greedy(model, reused, logits_reused, 32)

        # Reference arm: identical computation, no reuse machinery.
        fresh = make_prompt_cache(model)
        prefill(model, fresh, prompt, chunk)
        replay(model, fresh, generated)
        logits_fresh = prefill(model, fresh, extra, chunk)
        cont_fresh = greedy(model, fresh, logits_fresh, 32)

        self.assertTrue(
            mx.allclose(
                logits_reused[:, -1, :], logits_fresh[:, -1, :], atol=1e-5
            ).item()
        )
        self.assertEqual(cont_reused, cont_fresh)

    def test_generation_roundtrip_token_exact_qwen3_next(self):
        self._generation_roundtrip(QWEN3_NEXT_CONFIG)

    def test_generation_roundtrip_token_exact_kimi_linear(self):
        self._generation_roundtrip(KIMI_LINEAR_CONFIG)


if __name__ == "__main__":
    unittest.main()
