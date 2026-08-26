# Copyright © 2026 MLX Contributors
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import qwen3_moe
from mlx_lm.tuner.trainer import grad_checkpoint


class TestQwen3MoeGradients(unittest.TestCase):
    def make_model(self):
        args = qwen3_moe.ModelArgs(
            model_type="qwen3_moe",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=4,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            moe_intermediate_size=16,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            head_dim=4,
            rope_theta=10_000.0,
            tie_word_embeddings=False,
            max_position_embeddings=128,
            norm_topk_prob=True,
        )
        return qwen3_moe.Model(args)

    def assert_backward_succeeds(self, use_checkpoint):
        mx.random.seed(7)
        model = self.make_model()
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        layer_type = type(model.layers[0])
        original_call = layer_type.__call__
        if use_checkpoint:
            grad_checkpoint(model.layers[0])

        try:

            def loss_fn(current_model, inputs):
                return current_model(inputs).mean()

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, gradients = loss_and_grad(model, tokens)
            mx.eval(loss, gradients)
            self.assertTrue(mx.isfinite(loss).item())
        finally:
            layer_type.__call__ = original_call

    def test_backward(self):
        self.assert_backward_succeeds(use_checkpoint=False)

    def test_backward_with_gradient_checkpointing(self):
        self.assert_backward_succeeds(use_checkpoint=True)


if __name__ == "__main__":
    unittest.main()
