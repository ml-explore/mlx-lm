# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

from mlx_lm.models import llama
from mlx_lm.quant.dwq import dwq_quantize_layerwise


class TestLayerwiseDWQ(unittest.TestCase):
    def test_layerwise_rounds_run_and_preserve_quality(self):
        mx.random.seed(0)
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=128,
        )
        model = llama.Model(args)
        nn.quantize(model, group_size=32, bits=4)

        vocab = args.vocab_size
        seq = 16

        def make_data(n):
            data = []
            for _ in range(n):
                toks = mx.random.randint(0, vocab, (seq + 1,)).tolist()
                data.append((toks, 0))
            return data

        train_data = make_data(8)
        valid_data = make_data(2)

        # Fixed random teacher logits keep the loss well-defined without a
        # second model.
        def target_fn(batch, idx, split):
            mx.random.seed(hash((idx, split)) % (2**31))
            return mx.random.normal((*batch.shape, vocab))

        opt = optimizers.Adam(learning_rate=1e-4, bias_correction=True)
        dwq_quantize_layerwise(
            model,
            target_fn,
            opt,
            train_data,
            valid_data,
            batch_size=2,
            max_seq_length=seq,
            seed=0,
            layers_per_round=2,
        )

        # The model still produces finite logits after the rounds.
        out = model(mx.array([train_data[0][0][:-1]]))
        self.assertTrue(mx.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
