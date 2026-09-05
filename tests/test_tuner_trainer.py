# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.tuner.trainer import iterate_batches


class MockDistributedGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class TestTunerTrainer(unittest.TestCase):
    def test_iterate_batches_ddp(self):
        group = MockDistributedGroup(0, 1)

        def run(rank, size, batch):
            group._rank = rank
            group._size = size

            data = mx.arange(128).reshape(-1, 1).tolist()
            data = [(d, 0) for d in data]

            samples = set()
            for i, (b, l) in enumerate(
                iterate_batches(data, batch, 1, comm_group=group)
            ):
                samples.add(tuple(mx.flatten(b).tolist()))

            ref_batches = mx.arange(128).reshape(-1, batch).tolist()
            for b in ref_batches:
                self.assertTrue(tuple(b[rank::size]) in samples)

        run(0, 1, 4)
        run(0, 1, 8)
        run(0, 2, 8)
        run(1, 2, 8)
        run(0, 4, 8)
        run(1, 4, 8)
        run(2, 4, 8)
        run(3, 4, 8)

    def test_iterate_batches_seed(self):
        # One distinct token id per row so the batch order is observable.
        data = [([i + 1] * 8, 0) for i in range(64)]

        def order(seed, consume=0):
            np.random.seed(1)
            if consume:
                # Stand-in for anything else drawing from numpy in between.
                np.random.rand(consume)
            batches = iterate_batches(data, 4, 8, loop=True, seed=seed)
            return [b[0].tolist()[0] for b, _ in zip(batches, range(5))]

        # seed=0 must be honored like any other seed. It is also the default
        # in mlx_lm.lora's CONFIG_DEFAULTS, so `if seed:` silently dropped it.
        for seed in (0, 42):
            with self.subTest(seed=seed):
                self.assertEqual(order(seed), order(seed, consume=3))

        self.assertNotEqual(order(0), order(42))

    def _assert_backward(self, model):
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        def loss_fn(current_model, inputs):
            return current_model(inputs).mean()

        loss, gradients = nn.value_and_grad(model, loss_fn)(model, tokens)
        mx.eval(loss, gradients)
        self.assertTrue(mx.isfinite(loss).item())

    def test_qwen3_moe_backward(self):
        from mlx_lm.models import qwen3_moe

        model = qwen3_moe.Model(
            qwen3_moe.ModelArgs(
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
        )
        self._assert_backward(model)

    def test_granitemoe_backward(self):
        from mlx_lm.models import granitemoe

        model = granitemoe.Model(
            granitemoe.ModelArgs(
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
        )
        self._assert_backward(model)

    def test_granitemoehybrid_backward(self):
        from mlx_lm.models import granitemoehybrid

        model = granitemoehybrid.Model(
            granitemoehybrid.ModelArgs(
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
        )
        self._assert_backward(model)

    def test_lfm2_moe_backward(self):
        from mlx_lm.models import lfm2_moe

        model = lfm2_moe.Model(
            lfm2_moe.ModelArgs(
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
        )
        self._assert_backward(model)


if __name__ == "__main__":
    unittest.main()
