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
            {
                "model_type": "qwen3_moe",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [0],
                "norm_topk_prob": True,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "max_position_embeddings": 256,
                "tie_word_embeddings": False,
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

    def test_pipeline(self):
        test_configs = [
            {
                "model_type": "qwen3_moe",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [0],
                "norm_topk_prob": True,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "max_position_embeddings": 256,
                "tie_word_embeddings": False,
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
                model.model.pipeline(mx.distributed.init())
                out = model(x)
                self.assertTrue(mx.allclose(expected, out, rtol=1e-3, atol=1e-3))

    def test_pipeline_layer_assignment(self):
        from mlx_lm.models.pipeline import PipelineMixin

        class Group:
            def __init__(self, rank, size):
                self._rank, self._size = rank, size

            def rank(self):
                return self._rank

            def size(self):
                return self._size

        class Stage(PipelineMixin):
            def __init__(self, num_layers):
                super().__init__()
                self.layers = list(range(num_layers))

        def assignment(num_layers, size, split=None):
            blocks = []
            for rank in range(size):
                stage = Stage(num_layers)
                stage.pipeline(Group(rank, size), split=split)
                blocks.append(stage.pipeline_layers)
            return blocks

        # Every layer must run on exactly one rank, also when the layer count
        # does not divide evenly. The uneven cases dropped layers before.
        for num_layers, size in [(4, 2), (7, 2), (10, 4), (48, 3), (62, 4), (5, 5)]:
            blocks = assignment(num_layers, size)
            assigned = sorted(l for block in blocks for l in block)
            self.assertEqual(assigned, list(range(num_layers)), (num_layers, size))
            # Stage order is reverse rank order: the last rank runs the first
            # layers and rank 0 runs the last layers.
            self.assertEqual(blocks[size - 1][0], 0)
            self.assertEqual(blocks[0][-1], num_layers - 1)
            for block in blocks:
                self.assertEqual(block, list(range(block[0], block[-1] + 1)))

        # An explicit split sets the layer count per rank.
        blocks = assignment(4, 2, split=[1, 3])
        self.assertEqual(blocks[0], [3])
        self.assertEqual(blocks[1], [0, 1, 2])
        blocks = assignment(10, 3, split=[2, 3, 5])
        self.assertEqual([len(b) for b in blocks], [2, 3, 5])
        self.assertEqual(sorted(l for b in blocks for l in b), list(range(10)))

        # A bad split is an error, not a silent bad assignment.
        with self.assertRaises(ValueError):
            assignment(4, 2, split=[1, 1, 2])
        with self.assertRaises(ValueError):
            assignment(4, 2, split=[0, 4])
        with self.assertRaises(ValueError):
            assignment(4, 2, split=[2, 3])

    def test_pipeline_uneven_model(self):
        # 7 layers on 2 ranks: layer 3 ran on no rank before this fix.
        config = {
            "model_type": "qwen3_moe",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 7,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [0],
            "norm_topk_prob": True,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
        }

        class Group:
            def __init__(self, rank, size):
                self._rank, self._size = rank, size

            def rank(self):
                return self._rank

            def size(self):
                return self._size

        arch = importlib.import_module("mlx_lm.models.qwen3_moe")
        args = arch.ModelArgs.from_dict(config)
        assigned = []
        for rank in range(2):
            model = arch.Model(args)
            model.model.pipeline(Group(rank, 2))
            assigned.extend(
                i for i, l in enumerate(model.model.layers) if l is not None
            )
        self.assertEqual(sorted(assigned), list(range(7)))


if __name__ == "__main__":
    unittest.main()
