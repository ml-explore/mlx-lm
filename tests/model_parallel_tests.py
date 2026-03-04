# Copyright © 2026 Apple Inc.

import importlib
import unittest

import mlx.core as mx

import mlx_lm


class TestModelParallel(unittest.TestCase):

    def test_shard(self):
        test_configs = [
            {
                "model_type": "deepseek_v3",
                "vocab_size": 1024,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_group": 2,
                "topk_group": 1,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "kv_lora_rank": 4,
                "q_lora_rank": 4,
                "qk_rope_head_dim": 32,
                "v_head_dim": 16,
                "qk_nope_head_dim": 32,
                "rope_scaling": {
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 40,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn",
                },
            },
            {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 256,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 128,
                "sliding_window": 4,
                "layer_types": [
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "tie_word_embeddings": False,
                "rope_theta": 10000.0,
            },
            {
                "model_type": "glm4_moe_lite",
                "vocab_size": 1000,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "routed_scaling_factor": 1.0,
                "kv_lora_rank": 8,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 16,
                "v_head_dim": 8,
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
                "n_group": 1,
                "topk_group": 1,
                "num_experts_per_tok": 2,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 256,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000,
                "rope_scaling": None,
                "attention_bias": False,
                "partial_rotary_factor": 1.0,
                "tie_word_embeddings": False,
                "num_nextn_predict_layers": 1,
            },
        ]
        mx.random.seed(0)
        for config in test_configs:
            model_type = config["model_type"]
            with self.subTest(f"Testing {model_type}", model_type=model_type):
                arch = importlib.import_module(f"mlx_lm.models.{model_type}")
                args = arch.ModelArgs.from_dict(config)
                model = arch.Model(args)
                vocab_size = args.vocab_size
                x = mx.random.randint(0, vocab_size, shape=(32, 4))
                expected = model(x)
                model.shard()
                out = model(x)
                self.assertTrue(mx.allclose(expected, out, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
