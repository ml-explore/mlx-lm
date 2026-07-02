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
            self.assertEqual(cache.trim_prompt_cache(c, chunk.size - m),
                             chunk.size - m)
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
                        mx.array_equal(a.keys[..., : a.offset, :],
                                       b.keys[..., : b.offset, :]))
                    self.assertTrue(
                        mx.array_equal(a.values[..., : a.offset, :],
                                       b.values[..., : b.offset, :]))

    def test_speculative_matches_vanilla(self):
        # With an unrelated (random) draft model the acceptance rate is ~0, so
        # every round exercises the rollback path; greedy speculative decoding
        # must still reproduce the vanilla greedy generation exactly.
        model = make_hybrid()
        mx.random.seed(1)
        draft = llama.Model(llama.ModelArgs.from_dict(LLAMA_ARGS))
        prompt = mx.random.randint(0, 1000, (16,), dtype=mx.uint32)
        n = 24

        vanilla = [
            int(tok) for tok, _ in generate_step(prompt, model, max_tokens=n)
        ]
        spec = [
            int(tok)
            for tok, _, _ in speculative_generate_step(
                prompt, model, draft, num_draft_tokens=3, max_tokens=n
            )
        ]
        self.assertEqual(vanilla, spec)

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


if __name__ == "__main__":
    unittest.main()
