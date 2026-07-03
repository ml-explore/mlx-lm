# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.qwen3_5 import MTPModule, TextModel, TextModelArgs


def tiny_args(**overrides):
    kwargs = dict(
        model_type="qwen3_5_moe_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        # >= 64 so the fused gated-delta step kernel's tiling stays valid
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    kwargs.update(overrides)
    return TextModelArgs(**kwargs)


def gdn_states_of(cache_list):
    return [
        (c[0], c[1])
        for c in cache_list
        if c is not None and not c.is_trimmable()
    ]


class TestQwen35MTP(unittest.TestCase):
    def test_rollback_replay_matches_sequential(self):
        """After a speculative verify forward of `block` tokens,
        rollback(keep=j) must leave GDN caches in the same state as a
        model that only ever saw the first j block tokens."""
        mx.random.seed(0)
        model = TextModel(tiny_args())
        prompt = mx.random.randint(0, 64, (1, 8))
        block = mx.random.randint(0, 64, (1, 4))

        for keep in range(0, 5):
            # Ground truth: sequential path that never saw rejected tokens.
            ref_cache = model.make_cache()
            model(prompt, cache=ref_cache)
            if keep > 0:
                model(block[:, :keep], cache=ref_cache)
            ref_states = gdn_states_of(ref_cache)

            # Speculative path: full-width verify, then rollback.
            cache = model.make_cache()
            model(prompt, cache=cache)
            sink = []
            model.model(block, cache=cache, gdn_sink=sink)
            model.rollback_speculative_cache(cache, sink, keep, block.shape[1])
            got_states = gdn_states_of(cache)

            for (rc, rs), (gc, gs) in zip(ref_states, got_states):
                if keep == 0 and gs is None:
                    self.assertIsNone(rs) if rs is None else self.assertTrue(
                        mx.allclose(rs, mx.zeros_like(rs)).item()
                    )
                    continue
                # Batched (verify-width) vs sequential matmuls differ by
                # kernel-scheduling numerics (~1e-3 fp32); a wrong slice or
                # stale state is orders of magnitude larger.
                self.assertTrue(
                    mx.allclose(rc, gc, atol=5e-3).item(),
                    f"conv state mismatch at keep={keep}",
                )
                self.assertTrue(
                    mx.allclose(rs, gs, atol=5e-3).item(),
                    f"ssm state mismatch at keep={keep}",
                )
            # KV caches must hold prompt + keep tokens.
            kv = next(c for c in cache if c.is_trimmable())
            self.assertEqual(kv.offset, 8 + keep)

    def test_mtp_step_shapes_and_chaining(self):
        mx.random.seed(0)
        model = TextModel(tiny_args())
        self.assertIsInstance(model.mtp, MTPModule)
        cache = model.make_cache()
        mtp_cache = model.make_mtp_cache()
        prompt = mx.random.randint(0, 64, (1, 8))
        hidden = model.model(prompt, cache=cache)
        logits, post = model.mtp_step(hidden[:, :-1], prompt[:, 1:], mtp_cache)
        self.assertEqual(logits.shape, (1, 7, 64))
        self.assertEqual(post.shape, (1, 7, 32))
        self.assertEqual(mtp_cache[0].offset, 7)
        # Recursive chaining on the MTP's own hidden.
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        logits2, _ = model.mtp_step(post[:, -1:], tok, mtp_cache)
        self.assertEqual(logits2.shape, (1, 1, 64))
        self.assertEqual(mtp_cache[0].offset, 8)
        mtp_cache[0].trim(1)
        self.assertEqual(mtp_cache[0].offset, 7)

    def test_mtp_module_dropped_without_weights(self):
        model = TextModel(tiny_args())
        # A converted checkpoint without mtp tensors: module must drop so
        # strict loading stays consistent.
        weights = {"model.norm.weight": mx.ones((32,))}
        model.sanitize(dict(weights))
        self.assertIsNone(model.mtp)

    def test_no_norm_shift_for_converted_mtp_checkpoints(self):
        """mtp.* presence alone must NOT trigger the raw-checkpoint norm
        shift: converted '-mtp' checkpoints have sanitized conv1d and
        already-shifted norms."""
        model = TextModel(tiny_args())
        norm = mx.ones((32,))
        weights = {
            "model.norm.weight": norm,
            "mtp.norm.weight": norm,
            # sanitized conv1d (last dim 1) => not a raw checkpoint
            "model.layers.0.linear_attn.conv1d.weight": mx.zeros((24, 4, 1)),
        }
        out = model.sanitize(weights)
        self.assertTrue(mx.array_equal(out["model.norm.weight"], norm).item())
        self.assertTrue(mx.array_equal(out["mtp.norm.weight"], norm).item())
        # Raw checkpoint (unsanitized conv1d) still shifts.
        model2 = TextModel(tiny_args())
        weights2 = {
            "model.norm.weight": norm,
            "mtp.norm.weight": norm,
            "model.layers.0.linear_attn.conv1d.weight": mx.zeros((24, 1, 4)),
        }
        out2 = model2.sanitize(weights2)
        self.assertTrue(
            mx.array_equal(out2["model.norm.weight"], norm + 1.0).item()
        )
        self.assertTrue(
            mx.array_equal(out2["mtp.norm.weight"], norm + 1.0).item()
        )


if __name__ == "__main__":
    unittest.main()
