"""Tests for Nemotron-H LatentMoE support (PR #992).

Tests the additions to nemotron_h.py:
- ModelArgs: moe_latent_size, layers_block_type normalization, time_step_limit defaults
- NemotronHMoE: latent projection forward pass
- Model.sanitize: MTP weight stripping
"""
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.nemotron_h import Model, ModelArgs, NemotronHMoE


class TestModelArgsLatentMoE(unittest.TestCase):
    """Test ModelArgs parsing for Nemotron Super config fields."""

    def _base_args(self, **overrides):
        """Minimal valid config for nemotron_h with MoE layers."""
        cfg = {
            "model_type": "nemotron_h",
            "vocab_size": 1000,
            "hidden_size": 128,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "max_position_embeddings": 1000,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attention_bias": False,
            "mamba_num_heads": 4,
            "mamba_head_dim": 32,
            "mamba_proj_bias": False,
            "ssm_state_size": 32,
            "conv_kernel": 4,
            "n_groups": 2,
            "time_step_min": 0.001,
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "use_bias": False,
            "use_conv_bias": True,
            "hybrid_override_pattern": ["M", "E", "*", "E"],
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
        }
        cfg.update(overrides)
        return ModelArgs(**cfg)

    def test_moe_latent_size_parsed(self):
        args = self._base_args(moe_latent_size=32)
        self.assertEqual(args.moe_latent_size, 32)

    def test_moe_latent_size_none_by_default(self):
        args = self._base_args()
        self.assertIsNone(args.moe_latent_size)

    def test_layers_block_type_normalization(self):
        """layers_block_type (word list) should normalize to hybrid_override_pattern (char list)."""
        args = self._base_args(
            hybrid_override_pattern=None,
            layers_block_type=["mamba", "moe", "attention", "moe"],
        )
        self.assertEqual(args.hybrid_override_pattern, ["M", "E", "*", "E"])
        self.assertEqual(args.num_hidden_layers, 4)

    def test_hybrid_override_pattern_string(self):
        """Config from HuggingFace comes as a string, should work with iteration."""
        args = self._base_args(hybrid_override_pattern="ME*E")
        # String iterates as chars, len() returns 4
        self.assertEqual(len(args.hybrid_override_pattern), 4)
        self.assertEqual(list(args.hybrid_override_pattern), ["M", "E", "*", "E"])

    def test_time_step_limit_no_upper_bound(self):
        """time_step_limit should use inf upper bound when only time_step_min is set."""
        args = self._base_args(time_step_min=0.001)
        self.assertEqual(args.time_step_limit[0], 0.001)
        self.assertEqual(args.time_step_limit[1], float("inf"))

    def test_time_step_limit_explicit_overrides(self):
        """Explicit time_step_limit should not be overwritten."""
        args = self._base_args(time_step_limit=(0.01, 0.5), time_step_min=0.001)
        self.assertEqual(args.time_step_limit, (0.01, 0.5))


class TestNemotronHMoELatent(unittest.TestCase):
    """Test NemotronHMoE forward pass with latent projection."""

    def _make_config(self, moe_latent_size=None):
        return self._base_args(moe_latent_size=moe_latent_size)

    def _base_args(self, **overrides):
        cfg = {
            "model_type": "nemotron_h",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attention_bias": False,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "mamba_proj_bias": False,
            "ssm_state_size": 16,
            "conv_kernel": 4,
            "n_groups": 2,
            "time_step_min": 0.001,
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "use_bias": False,
            "use_conv_bias": True,
            "hybrid_override_pattern": ["E", "E"],
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": True,
        }
        cfg.update(overrides)
        return ModelArgs(**cfg)

    def test_latent_projection_shapes(self):
        """With moe_latent_size, experts should operate on latent dim."""
        config = self._make_config(moe_latent_size=16)
        moe = NemotronHMoE(config)
        mx.eval(moe.parameters())

        # Input: (batch=1, seq=1, hidden=64)
        x = mx.random.normal((1, 1, 64))
        y = moe(x)
        mx.eval(y)

        # Output should match hidden_size, not latent size
        self.assertEqual(y.shape, (1, 1, 64))

    def test_no_latent_projection(self):
        """Without moe_latent_size, experts operate at full hidden dim."""
        config = self._make_config(moe_latent_size=None)
        moe = NemotronHMoE(config)
        mx.eval(moe.parameters())

        x = mx.random.normal((1, 1, 64))
        y = moe(x)
        mx.eval(y)
        self.assertEqual(y.shape, (1, 1, 64))

    def test_latent_projection_has_layers(self):
        """LatentMoE should have fc1/fc2 latent projection layers."""
        config = self._make_config(moe_latent_size=16)
        moe = NemotronHMoE(config)
        self.assertTrue(hasattr(moe, "fc1_latent_proj"))
        self.assertTrue(hasattr(moe, "fc2_latent_proj"))
        # fc1: hidden(64) -> latent(16)
        self.assertEqual(moe.fc1_latent_proj.weight.shape, (16, 64))
        # fc2: latent(16) -> hidden(64)
        self.assertEqual(moe.fc2_latent_proj.weight.shape, (64, 16))

    def test_shared_expert_gets_original_input(self):
        """Shared expert should receive the original residuals, not latent-projected input."""
        config = self._make_config(moe_latent_size=16)
        config.n_shared_experts = 1
        config.moe_shared_expert_intermediate_size = 32
        moe = NemotronHMoE(config)
        mx.eval(moe.parameters())

        x = mx.random.normal((1, 1, 64))
        # Just verify it runs without error — the shared expert
        # should accept hidden_size(64) input, not latent_size(16)
        y = moe(x)
        mx.eval(y)
        self.assertEqual(y.shape, (1, 1, 64))


class TestSanitizeMTP(unittest.TestCase):
    """Test that sanitize() strips MTP weights."""

    def test_mtp_weights_stripped(self):
        config = ModelArgs(
            model_type="nemotron_h",
            vocab_size=100,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=2,
            max_position_embeddings=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=False,
            mamba_num_heads=4,
            mamba_head_dim=16,
            mamba_proj_bias=False,
            ssm_state_size=16,
            conv_kernel=4,
            n_groups=2,
            time_step_min=0.001,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            hybrid_override_pattern=["*", "M"],
        )
        model = Model(config)
        weights = {
            "model.embed_tokens.weight": mx.zeros((100, 64)),
            "model.layers.0.norm.weight": mx.zeros((64,)),
            "mtp.layers.0.weight": mx.zeros((64, 64)),
            "mtp.head.weight": mx.zeros((100, 64)),
        }
        sanitized = model.sanitize(weights)
        # MTP weights should be removed
        self.assertNotIn("mtp.layers.0.weight", sanitized)
        self.assertNotIn("mtp.head.weight", sanitized)
        # Non-MTP weights should be preserved
        self.assertIn("model.embed_tokens.weight", sanitized)
        self.assertIn("model.layers.0.norm.weight", sanitized)


if __name__ == "__main__":
    unittest.main()
