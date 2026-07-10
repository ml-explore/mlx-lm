# Copyright © 2024 Apple Inc.

import unittest

import mlx.nn as nn

from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache


class MakeCacheAcceptsMaxKV(nn.Module):
    """Stub model whose make_cache accepts max_kv_size (hybrid-style)."""

    def __init__(self):
        super().__init__()
        self.received = "unset"

    def make_cache(self, max_kv_size=None):
        self.received = max_kv_size
        return [KVCache()]


class MakeCacheNoMaxKV(nn.Module):
    """Stub model whose make_cache does not accept max_kv_size (legacy)."""

    def __init__(self):
        super().__init__()
        self.called = False

    def make_cache(self):
        self.called = True
        return [KVCache()]


class TestMakePromptCacheMaxKV(unittest.TestCase):
    def test_forwards_max_kv_size_when_accepted(self):
        model = MakeCacheAcceptsMaxKV()
        make_prompt_cache(model, max_kv_size=128)
        # The fix must forward the budget to models that accept it.
        self.assertEqual(model.received, 128)

    def test_no_forward_when_not_accepted(self):
        model = MakeCacheNoMaxKV()
        # Legacy models without a max_kv_size parameter must still work and be
        # called with no arguments (backward compatible).
        cache = make_prompt_cache(model, max_kv_size=128)
        self.assertTrue(model.called)
        self.assertEqual(len(cache), 1)

    def test_no_forward_when_budget_is_none(self):
        model = MakeCacheAcceptsMaxKV()
        make_prompt_cache(model)
        self.assertIsNone(model.received)

    def test_qwen3_next_caps_attention_layers(self):
        from mlx_lm.models import qwen3_next

        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=4,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=2,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=64,
            mlp_only_layers=[],
            moe_intermediate_size=64,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            max_position_embeddings=64,
            head_dim=16,
            full_attention_interval=2,
        )
        model = qwen3_next.Model(args)

        capped = make_prompt_cache(model, max_kv_size=128)
        uncapped = make_prompt_cache(model)

        linear_flags = [l.is_linear for l in model.layers]
        # Attention (non-linear) layers must be rotating when a budget is set,
        # recurrent (linear) layers stay unbounded either way.
        self.assertTrue(any(not f for f in linear_flags))
        for layer_cache, is_linear in zip(capped, linear_flags):
            if not is_linear:
                self.assertIsInstance(layer_cache, RotatingKVCache)
        for layer_cache, is_linear in zip(uncapped, linear_flags):
            if not is_linear:
                self.assertIsInstance(layer_cache, KVCache)


if __name__ == "__main__":
    unittest.main()
