# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models import granitemoehybrid as G


class TestGraniteMoeHybridPaddedBatch(unittest.TestCase):
    def test_padded_batch_logprobs_match_alone(self):
        # Regression for #1908: shorter prompt in a left-padded batch must
        # match standalone first-step logprobs. Needs both:
        # - ArraysCache.make_mask ANDing left_padding with lengths
        # - GraniteMoeHybrid _ssm passing cache.lengths into ssm_update
        mx.random.seed(0)
        model = G.Model(
            G.ModelArgs(
                model_type="granitemoehybrid",
                vocab_size=100,
                hidden_size=64,
                intermediate_size=128,
                shared_intermediate_size=128,
                num_hidden_layers=2,
                layer_types=["mamba", "attention"],
                max_position_embeddings=512,
                num_attention_heads=4,
                num_key_value_heads=2,
                attention_bias=False,
                embedding_multiplier=1.0,
                attention_multiplier=0.25,
                logits_scaling=1.0,
                residual_multiplier=1.0,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                position_embedding_type="nope",
                mamba_n_heads=4,
                mamba_d_head=32,
                mamba_d_state=32,
                mamba_d_conv=4,
                mamba_n_groups=1,
                mamba_proj_bias=False,
                mamba_conv_bias=True,
            )
        )
        # Random init under-weights the recurrent path; amplify like the issue.
        for layer in model.layers:
            if hasattr(layer, "mamba"):
                layer.mamba.A_log = mx.full(layer.mamba.A_log.shape, -4.0)
                layer.mamba.conv1d.weight = layer.mamba.conv1d.weight * 30
        mx.eval(model.parameters())

        long_prompt = list(range(1, 13))
        short_prompt = [7, 8, 9]

        _, alone = next(
            iter(generate_step(mx.array(short_prompt), model, max_tokens=1))
        )

        gen = BatchGenerator(model, max_tokens=1, stop_tokens=None)
        uids = gen.insert([long_prompt, short_prompt])
        got = {}
        while len(got) < 2:
            for r in gen.next_generated():
                got.setdefault(r.uid, r.logprobs)
        gen.close()

        err = mx.abs(got[uids[1]] - alone).max().item()
        self.assertLess(
            err, 1e-5, f"padded vs alone max |Δlogprob| = {err:.3e}"
        )


if __name__ == "__main__":
    unittest.main()
