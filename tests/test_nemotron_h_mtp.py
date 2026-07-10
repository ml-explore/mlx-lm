# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.nemotron_h import Model, ModelArgs, NemotronHMTP


def tiny_args(**overrides):
    kwargs = dict(
        model_type="nemotron_h",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        max_position_embeddings=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=16,
        mamba_proj_bias=False,
        ssm_state_size=16,
        conv_kernel=3,
        n_groups=2,
        time_step_limit=(0.0, float("inf")),
        mlp_bias=False,
        layer_norm_epsilon=1e-4,
        use_bias=True,
        use_conv_bias=True,
        # attention + mamba backbone
        hybrid_override_pattern=["*", "M", "*", "M"],
        # DeepSeek-style MTP head: first layer fuses embed/hidden (eh_proj /
        # enorm / hnorm), last carries final_layernorm. The shipped Nemotron-3
        # Super head is transformer (attention) blocks; mtp_step builds the
        # attention mask accordingly.
        num_nextn_predict_layers=2,
        mtp_layers_block_type=["attention", "attention"],
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


def _kv_offset(mtp_cache):
    from mlx_lm.models.cache import KVCache

    return next(c.offset for c in mtp_cache if isinstance(c, KVCache))


class TestNemotronHMTP(unittest.TestCase):
    def test_mtp_step_shapes_and_chaining(self):
        mx.random.seed(0)
        model = Model(tiny_args())
        self.assertIsInstance(model.mtp, NemotronHMTP)
        cache = model.make_cache()
        mtp_cache = model.make_mtp_cache()
        prompt = mx.random.randint(0, 64, (1, 8))
        hidden = model.backbone(prompt, cache=cache)
        logits, post = model.mtp_step(hidden[:, :-1], prompt[:, 1:], mtp_cache)
        self.assertEqual(logits.shape, (1, 7, 64))
        self.assertEqual(post.shape, (1, 7, 32))
        self.assertEqual(_kv_offset(mtp_cache), 7)
        # The head produces a real next-token prediction.
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        self.assertEqual(tok.shape, (1, 1))
        self.assertTrue((0 <= tok).all().item() and (tok < 64).all().item())
        # Recursive chaining on the MTP's own hidden.
        logits2, _ = model.mtp_step(post[:, -1:], tok, mtp_cache)
        self.assertEqual(logits2.shape, (1, 1, 64))
        self.assertEqual(_kv_offset(mtp_cache), 8)

    def test_mtp_module_dropped_without_weights(self):
        model = Model(tiny_args())
        # A checkpoint without mtp tensors: module must drop so strict loading
        # stays consistent.
        weights = {"backbone.norm_f.weight": mx.ones((32,))}
        model.sanitize(dict(weights))
        self.assertIsNone(model.mtp)

    def test_rollback_speculative_cache_matches_fresh_forward(self):
        """A verify forward of `block_size` tokens, of which `keep` are
        accepted, must leave the (KV + Mamba) caches identical to a fresh
        forward over just the committed prefix + kept tokens. Exercises the
        ssm_sink capture and rollback_speculative_cache replay."""
        mx.random.seed(0)
        model = Model(tiny_args())
        tokens = mx.random.randint(0, 64, (1, 12))
        prefix, block_size, keep = 6, 4, 2

        # Ground truth: fresh forward over prefix + kept tokens.
        ref = model.make_cache()
        model.backbone(tokens[:, : prefix + keep], cache=ref)

        # Speculative: commit prefix, verify a block, then roll back.
        spec = model.make_cache()
        model.backbone(tokens[:, :prefix], cache=spec)
        sink = []
        model.backbone(
            tokens[:, prefix : prefix + block_size], cache=spec, ssm_sink=sink
        )
        # One ssm capture per Mamba layer in the backbone.
        n_mamba = sum(1 for l in model.layers if l.block_type == "M")
        self.assertEqual(len(sink), n_mamba)
        model.rollback_speculative_cache(spec, sink, keep, block_size)

        for cr, cs in zip(ref, spec):
            if cr.is_trimmable():  # KV cache
                self.assertEqual(cr.offset, cs.offset)
                self.assertTrue(
                    mx.allclose(
                        cr.keys[..., : cr.offset, :],
                        cs.keys[..., : cs.offset, :],
                        atol=1e-4,
                    ).item()
                )
            else:  # Mamba ArraysCache: compare recurrent state
                self.assertTrue(mx.allclose(cr[1], cs[1], atol=1e-4).item())


if __name__ == "__main__":
    unittest.main()
