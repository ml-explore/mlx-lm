# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models import deepseek_v41
from mlx_lm.models.rope_utils import YarnRoPE


class TestDeepseekV41(unittest.TestCase):
    def test_rope_inverse(self):
        rope = YarnRoPE(
            dims=4,
            traditional=True,
            base=160000.0,
            scaling_factor=2.0,
            original_max_position_embeddings=16,
            mscale=0.0,
            mscale_all_dim=0.0,
        )
        for shape in ((1, 5, 8), (1, 5, 2, 8)):
            x = mx.random.normal(shape)
            y = deepseek_v41._apply_rope(x, rope, rope_dim=4, offset=4, scale=2.0)
            y = deepseek_v41._apply_rope(
                y, rope, rope_dim=4, offset=4, scale=2.0, inverse=True
            )
            self.assertTrue(mx.allclose(y, x, atol=1e-5))

    def test_cached_matches_forward(self):
        args = deepseek_v41.ModelArgs(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            head_dim=32,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            o_groups=2,
            o_lora_rank=16,
            moe_intermediate_size=32,
            n_routed_experts=8,
            num_experts_per_tok=2,
            sliding_window=4,
            compress_ratios=(0, 2, 2, 2, 1, 1),
            kv_source_layer_ids=(1, 4),
            index_source_layer_ids=(1, 2, 4, 5),
            index_n_heads=4,
            index_head_dim=32,
            index_topk=3,
            candidate_source_layer_id=4,
            candidate_topk_blocks=2,
            candidate_block_size=2,
        )
        mx.random.seed(0)
        model = deepseek_v41.Model(args)
        tokens = mx.random.randint(0, args.vocab_size, (2, 16))
        full = model(tokens)

        cache = model.make_cache()
        steps = []
        for start, end in ((0, 7), (7, 15), (15, 16)):
            steps.append(model(tokens[:, start:end], cache=cache))
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
