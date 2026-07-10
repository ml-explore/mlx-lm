# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.qwen3_next import Model, ModelArgs, Qwen3NextMTP


def tiny_args(**overrides):
    kwargs = dict(
        model_type="qwen3_next",
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
        decoder_sparse_step=1,
        shared_expert_intermediate_size=16,
        mlp_only_layers=[],
        moe_intermediate_size=16,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=131072,
        mtp_num_hidden_layers=1,
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


class TestQwen3NextMTP(unittest.TestCase):
    def test_mtp_step_shapes_and_chaining(self):
        mx.random.seed(0)
        model = Model(tiny_args())
        self.assertIsInstance(model.mtp, Qwen3NextMTP)
        cache = model.make_cache()
        mtp_cache = model.make_mtp_cache()
        prompt = mx.random.randint(0, 64, (1, 8))
        # Trunk forward -> post-final-norm hidden states.
        hidden = model.model(prompt, cache=cache)
        self.assertEqual(hidden.shape, (1, 8, 32))
        # MTP predicts token p+2 from hidden_p and committed token p+1.
        logits, post = model.mtp_step(hidden[:, :-1], prompt[:, 1:], mtp_cache)
        self.assertEqual(logits.shape, (1, 7, 64))
        self.assertEqual(post.shape, (1, 7, 32))
        self.assertEqual(mtp_cache[0].offset, 7)
        # The head produces a valid next-token prediction (finite logits).
        self.assertTrue(mx.all(mx.isfinite(logits)).item())
        # Recursive chaining on the MTP's own hidden.
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        logits2, _ = model.mtp_step(post[:, -1:], tok, mtp_cache)
        self.assertEqual(logits2.shape, (1, 1, 64))
        self.assertEqual(mtp_cache[0].offset, 8)
        mtp_cache[0].trim(1)
        self.assertEqual(mtp_cache[0].offset, 7)

    def test_mtp_module_dropped_without_weights(self):
        model = Model(tiny_args())
        # A converted checkpoint without mtp tensors: module must drop so
        # strict loading stays consistent.
        weights = {"model.norm.weight": mx.ones((32,))}
        model.sanitize(dict(weights))
        self.assertIsNone(model.mtp)

    def test_mtp_weights_kept_when_present(self):
        model = Model(tiny_args())
        # mtp.* tensors present AND module built: keep them, keep the module.
        weights = {
            "model.norm.weight": mx.ones((32,)),
            "mtp.norm.weight": mx.ones((32,)),
        }
        out = model.sanitize(dict(weights))
        self.assertIsInstance(model.mtp, Qwen3NextMTP)
        self.assertIn("mtp.norm.weight", out)


if __name__ == "__main__":
    unittest.main()
