# Copyright © 2026 MLX Contributors
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import granitemoe, granitemoehybrid, lfm2_moe


def make_granitemoe():
    args = granitemoe.ModelArgs(
        model_type="granitemoe",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=32,
        logits_scaling=1.0,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        max_position_embeddings=128,
        num_key_value_heads=2,
        attention_bias=False,
        rope_theta=10_000.0,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return granitemoe.Model(args)


def make_granitemoehybrid():
    args = granitemoehybrid.ModelArgs(
        model_type="granitemoehybrid",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        max_position_embeddings=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        embedding_multiplier=1.0,
        attention_multiplier=1.0,
        logits_scaling=1.0,
        residual_multiplier=1.0,
        layer_types=["attention"],
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=32,
    )
    return granitemoehybrid.Model(args)


def make_lfm2_moe():
    args = lfm2_moe.ModelArgs(
        model_type="lfm2_moe",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_expert_bias=False,
        num_dense_layers=1,
        norm_eps=1e-6,
        conv_bias=False,
        conv_L_cache=3,
        layer_types=["conv", "full_attention"],
    )
    return lfm2_moe.Model(args)


class TestMoeRoutingGradients(unittest.TestCase):
    """Backward through MoE routers must not request a VJP w.r.t. the integer
    top-k expert indices (gather_axis has no VJP for indices)."""

    def assert_backward_succeeds(self, model):
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        def loss_fn(current_model, inputs):
            return current_model(inputs).mean()

        loss, gradients = nn.value_and_grad(model, loss_fn)(model, tokens)
        mx.eval(loss, gradients)
        self.assertTrue(mx.isfinite(loss).item())

    def test_granitemoe_backward(self):
        mx.random.seed(7)
        self.assert_backward_succeeds(make_granitemoe())

    def test_granitemoehybrid_backward(self):
        mx.random.seed(7)
        self.assert_backward_succeeds(make_granitemoehybrid())

    def test_lfm2_moe_backward(self):
        mx.random.seed(7)
        self.assert_backward_succeeds(make_lfm2_moe())


if __name__ == "__main__":
    unittest.main()
