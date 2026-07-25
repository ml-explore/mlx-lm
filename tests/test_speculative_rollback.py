# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models import cache, llama, qwen3_next

QWEN3_NEXT_ARGS = {
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

LLAMA_ARGS = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 1000,
}


def make_hybrid(seed=0):
    mx.random.seed(seed)
    return qwen3_next.Model(qwen3_next.ModelArgs.from_dict(QWEN3_NEXT_ARGS))


class TestSpeculativeRollback(unittest.TestCase):

    def test_rollback_is_exact(self):
        # Tail independence: feed two verify chunks sharing the first m tokens
        # but with different tails, roll both back to m — every cache entry and
        # the next-token logits must be IDENTICAL (no numeric tolerance: same
        # starting state, same shapes, deterministic kernels). Any off-by-one
        # in the recorded rollback leaks the divergent tail.
        model = make_hybrid()
        prompt = mx.random.randint(0, 1000, (32,), dtype=mx.uint32)
        T = 8
        xs = mx.random.randint(0, 1000, (T,), dtype=mx.uint32)
        zs = mx.random.randint(0, 1000, (T,), dtype=mx.uint32)
        probe = mx.array([7], mx.uint32)

        def run(chunk, m):
            c = model.make_cache()
            model(prompt[None], cache=c)
            mx.eval([x.state for x in c])
            for x in c:
                x.start_speculation()
            model(chunk[None], cache=c)
            self.assertEqual(cache.trim_prompt_cache(c, chunk.size - m), chunk.size - m)
            logits = model(probe[None], cache=c)
            mx.eval(logits, [x.state for x in c])
            return c, logits

        for m in (1, 3, T - 1):
            cA, lA = run(xs, m)
            chunk_b = mx.concatenate([xs[:m], zs[: T - m]])
            cB, lB = run(chunk_b, m)
            self.assertTrue(mx.array_equal(lA, lB))
            for a, b in zip(cA, cB):
                if isinstance(a, cache.ArraysCache):
                    for ea, eb in zip(a.cache, b.cache):
                        self.assertTrue(mx.array_equal(ea, eb))
                else:
                    self.assertEqual(a.offset, b.offset)
                    self.assertTrue(
                        mx.array_equal(
                            a.keys[..., : a.offset, :], b.keys[..., : b.offset, :]
                        )
                    )
                    self.assertTrue(
                        mx.array_equal(
                            a.values[..., : a.offset, :], b.values[..., : b.offset, :]
                        )
                    )

    def test_speculative_matches_vanilla(self):
        # With an unrelated (random) draft model the acceptance rate is ~0, so
        # every round exercises the rollback path; greedy speculative decoding
        # must still reproduce the vanilla greedy generation exactly.
        model = make_hybrid()
        mx.random.seed(1)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)
        n = 24

        vanilla = [int(tok) for tok, _ in generate_step(prompt, model, max_tokens=n)]
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=n
            )
        ]
        self.assertEqual(vanilla, spec)

    def test_cache_reusable_after_speculation(self):
        # A prompt cache used for speculative decoding must come back clean:
        # not trimmable, no dangling rollback, and usable for plain decoding
        # that matches a never-speculated run.
        model = make_hybrid()
        mx.random.seed(2)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)

        c = cache.make_prompt_cache(model) + cache.make_prompt_cache(draft)
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=8, prompt_cache=c
            )
        ]
        model_cache = c[: len(model.layers)]
        self.assertFalse(cache.can_trim_prompt_cache(model_cache))
        # The whole cache list must come back clean, draft caches included:
        # they start speculating alongside the model cache and record during
        # draft steps, so a dangling speculating flag would make a reused
        # draft cache trim through stale rollback records.
        for x in c:
            self.assertFalse(getattr(x, "speculating", False))
            if isinstance(x, cache.ArraysCache):
                self.assertEqual(len(x._rollbacks), 0)

        # continue decoding on the same cache with plain steps
        cont = [
            int(tok)
            for tok, _ in generate_step(
                mx.array(spec[-1:], mx.uint32),
                model,
                max_tokens=8,
                prompt_cache=model_cache,
            )
        ]

        # reference: one uninterrupted vanilla run over the same tokens
        ref = [
            int(tok)
            for tok, _ in generate_step(prompt, model, max_tokens=8 + len(spec))
        ]
        self.assertEqual(ref, spec + cont)

    def test_early_generator_close(self):
        # Closing the speculative generator mid-stream must roll the cache back
        # cleanly (the finally path) without raising.
        model = make_hybrid()
        mx.random.seed(3)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)

        gen = speculative_generate_step(
            prompt, model, draft, num_draft_tokens=3, max_tokens=64
        )
        took = [next(gen) for _ in range(3)]
        gen.close()  # must not raise (double-rewind / consumed-rollback bugs)
        self.assertEqual(len(took), 3)

    def test_unsupported_hybrid_still_raises(self):
        # An ArraysCache whose layers never record a rollback must not be
        # silently trimmable outside of speculation.
        c = cache.ArraysCache(size=2)
        self.assertFalse(c.is_trimmable())
        c.start_speculation()
        self.assertTrue(c.is_trimmable())
        with self.assertRaises(RuntimeError):
            c.trim(1)
        c.stop_speculation()
        self.assertFalse(c.is_trimmable())

    def test_hybrid_draft_model(self):
        # Target AND draft are hybrid GDN models (the setup reported in #1446:
        # Qwen3.6 target + Qwen3.5 draft). The draft cache is rewound exactly
        # too — its rollback spans several T=1 records per round — and the
        # output must still equal vanilla greedy.
        model = make_hybrid(seed=0)
        draft = make_hybrid(seed=7)  # different weights, same arch
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)
        n = 24

        vanilla = [int(tok) for tok, _ in generate_step(prompt, model, max_tokens=n)]
        c = cache.make_prompt_cache(model) + cache.make_prompt_cache(draft)
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=n, prompt_cache=c
            )
        ]
        self.assertEqual(vanilla, spec)
        # Draft caches record too (T=1 records per draft step), so they must
        # be stopped alongside the model caches: a dangling speculating flag
        # would make a reused draft cache trim through stale rollback records.
        for x in c:
            self.assertFalse(getattr(x, "speculating", False))
            if isinstance(x, cache.ArraysCache):
                self.assertEqual(len(x._rollbacks), 0)

    def test_draft_exception_not_masked(self):
        # An exception raised mid-round (e.g. inside the draft model) must
        # surface as itself: the finally-block cache rewind must not replace it
        # with a rollback RuntimeError.
        model = make_hybrid()
        mx.random.seed(4)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)

        boom = ValueError("draft exploded")
        calls = {"n": 0}

        class Flaky:
            def __call__(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] > 4:  # fail during the second round's draft steps
                    raise boom
                return draft(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(draft, name)

        gen = speculative_generate_step(
            prompt, model, Flaky(), num_draft_tokens=3, max_tokens=32
        )
        with self.assertRaises(ValueError) as ctx:
            for _ in gen:
                pass
        self.assertIs(ctx.exception, boom)

    def test_rewind_refusal_raises_not_silent(self):
        # A cache can pass the support gate (record_rollback attr + the model
        # declaring support) while never actually arming: its layer never calls
        # record_rollback, so is_trimmable() stays False mid-speculation. Since
        # trim_prompt_cache is all-or-none, that would silently rewind NOTHING
        # in ANY layer and generation would continue over ghost tokens. The
        # rewind must fail loudly instead.
        class UnarmedArraysCache(cache.ArraysCache):
            def start_speculation(self):  # gate passes, recording never arms
                pass

        model = make_hybrid(seed=0)
        draft = make_hybrid(seed=7)
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)
        c = cache.make_prompt_cache(model) + cache.make_prompt_cache(draft)
        for i, x in enumerate(c):
            if isinstance(x, cache.ArraysCache):
                bad = UnarmedArraysCache(len(x.cache) if hasattr(x, "cache") else 2)
                bad.__dict__.update(x.__dict__)
                c[i] = bad
                break
        with self.assertRaises(RuntimeError):
            for _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=24,
                prompt_cache=c,
            ):
                pass

    def test_trim_beyond_window_raises(self):
        # Trimming more than the recorded rollback window must fail loudly and
        # consistently rather than silently clamping (which would desync the
        # ArraysCache layers from the KVCache layers).
        model = make_hybrid()
        prompt = mx.random.randint(0, 1000, (32,), dtype=mx.uint32)
        c = model.make_cache()
        model(prompt[None], cache=c)
        mx.eval([x.state for x in c])
        for x in c:
            x.start_speculation()
        chunk = mx.random.randint(0, 1000, (4,), dtype=mx.uint32)
        model(chunk[None], cache=c)
        arrays = next(x for x in c if isinstance(x, cache.ArraysCache))
        with self.assertRaises(RuntimeError):
            arrays.trim(10)  # only 4 tokens of rollback recorded


if __name__ == "__main__":
    unittest.main()
