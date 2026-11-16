# Copyright © 2026 Apple Inc.

import importlib
import unittest

import mlx.core as mx

from mlx_lm.models import (
    gemma4,
    gemma4_text,
    muse_spark,
    qwen2,
    qwen3_moe,
    qwen3_next,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.models.pipeline import PipelineMixin
from mlx_lm.utils import quantize_model

MUSE_SPARK_OPEN = {
    "model_type": "muse_spark",
    "text_config": {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "layer_rope_theta": [10000.0, 10000.0, 10000.0, 0],
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "sliding_window": 8,
        "final_logit_softcapping": 20.0,
        "output_multiplier": 1.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_hidden_size": 64,
        "moe_intermediate_size": 128,
    },
}


class Group:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def gemma4_text_config(**overrides):
    # Layers 2 and 5 are full attention, the others are sliding attention.
    config = {
        "model_type": "gemma4_text",
        "hidden_size": 64,
        "num_hidden_layers": 6,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "global_head_dim": 32,
        "vocab_size": 128,
        "vocab_size_per_layer_input": 128,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "sliding_window": 8,
        "sliding_window_pattern": 3,
    }
    config.update(overrides)
    return config


def make_gemma4_text(**overrides):
    mx.random.seed(0)
    args = gemma4_text.ModelArgs.from_dict(gemma4_text_config(**overrides))
    model = gemma4_text.Model(args)
    mx.eval(model.parameters())
    return model


GEMMA4_KV_SHARED = {"num_kv_shared_layers": 2, "hidden_size_per_layer_input": 16}

GEMMA4_MOE = {
    "enable_moe_block": True,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 32,
}


class Stage(PipelineMixin):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = list(range(num_layers))


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
            MUSE_SPARK_OPEN,
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
            {
                "model_type": "deepseek_v32",
                "vocab_size": 128,
                "hidden_size": 64,
                "index_head_dim": 16,
                "index_n_heads": 2,
                "index_topk": 4,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "kv_lora_rank": 8,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 8,
                "v_head_dim": 16,
                "qk_nope_head_dim": 16,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 256,
            },
            {
                "model_type": "qwen2",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "max_position_embeddings": 256,
                "tie_word_embeddings": False,
            },
            {
                "model_type": "qwen3_next",
                "vocab_size": 128,
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 32,
                "mlp_only_layers": [0],
                "moe_intermediate_size": 32,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "max_position_embeddings": 256,
                "full_attention_interval": 4,
            },
            MUSE_SPARK_OPEN,
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

    def test_muse_spark_decode(self):
        # Prefill uses the whole pooler; decode steps of one token use the
        # row-split pooler and unpooler.
        for mode in (None, "nvfp4"):
            with self.subTest(mode=mode):
                mx.random.seed(0)
                args = muse_spark.ModelArgs.from_dict(MUSE_SPARK_OPEN)
                model = muse_spark.Model(args)
                if mode is not None:
                    quantize_model(model, {}, None, None, mode=mode, global_scale=True)
                x = mx.random.randint(0, args.text.vocab_size, shape=(1, 13))

                def run():
                    cache = make_prompt_cache(model)
                    outputs = [model(x[:, :10], cache=cache)]
                    for i in range(10, 13):
                        outputs.append(model(x[:, i : i + 1], cache=cache))
                    return outputs

                expected = run()
                model.shard()
                for e, out in zip(expected, run()):
                    self.assertTrue(mx.allclose(e, out, rtol=1e-3, atol=1e-3))

    def test_gemma4_pipeline_layers_and_cache(self):
        # Rank 0 runs the last layers. KV shared layers (4, 5) have no cache.
        cases = [
            ({}, None, {0: [3, 4, 5], 1: [0, 1, 2]}, {0: 3, 1: 3}),
            (GEMMA4_KV_SHARED, [4, 2], {0: [2, 3, 4, 5], 1: [0, 1]}, {0: 2, 1: 2}),
            (GEMMA4_KV_SHARED, [5, 1], {0: [1, 2, 3, 4, 5], 1: [0]}, {0: 3, 1: 1}),
        ]
        for overrides, split, layers, caches in cases:
            for rank in (0, 1):
                with self.subTest(overrides=overrides, split=split, rank=rank):
                    model = make_gemma4_text(**overrides)
                    model.model.pipeline(Group(rank, 2), split=split)
                    local = model.model.pipeline_layers
                    self.assertEqual(
                        [l.self_attn.layer_idx for l in local], layers[rank]
                    )
                    cache = model.make_cache()
                    self.assertEqual(len(cache), caches[rank])
                    for c, l in zip(cache, local):
                        expected = (
                            KVCache
                            if l.layer_type == "full_attention"
                            else RotatingKVCache
                        )
                        self.assertIsInstance(c, expected)

    def test_gemma4_pipeline_split_rejection(self):
        # Layers 4 and 5 read the KV of layers 3 and 2, so a split must keep
        # each shared layer on the rank of its source.
        for split in ([3, 3], [2, 4], [1, 5]):
            with self.subTest(split=split):
                model = make_gemma4_text(**GEMMA4_KV_SHARED)
                with self.assertRaisesRegex(ValueError, "KV shared"):
                    model.model.pipeline(Group(0, 2), split=split)
        for split in ([4, 2], [5, 1]):
            with self.subTest(split=split):
                model = make_gemma4_text(**GEMMA4_KV_SHARED)
                model.model.pipeline(Group(0, 2), split=split)

        # Without KV sharing every split is valid.
        for split in ([3, 3], [1, 5]):
            with self.subTest(split=split, shared=False):
                make_gemma4_text().model.pipeline(Group(0, 2), split=split)

    def test_gemma4_pipeline(self):
        for name, overrides in (("plain", {}), ("kv_shared", GEMMA4_KV_SHARED)):
            with self.subTest(name):
                model = make_gemma4_text(**overrides)
                x = mx.random.randint(0, 128, shape=(2, 7))
                expected = model(x)
                # A group of one does no send or recv.
                model.model.pipeline(mx.distributed.init())
                out = model(x)
                self.assertTrue(mx.allclose(expected, out, rtol=1e-3, atol=1e-3))

    def test_gemma4_wrapper_pipeline(self):
        config = {
            "model_type": "gemma4",
            "vocab_size": 128,
            "text_config": gemma4_text_config(),
        }
        args = gemma4.ModelArgs.from_dict(config)
        model = gemma4.Model(args)
        self.assertIs(model.model, model.language_model.model)
        self.assertEqual(len(model.layers), 6)
        for rank, expected in ((0, [3, 4, 5]), (1, [0, 1, 2])):
            with self.subTest(rank=rank):
                model = gemma4.Model(args)
                model.model.pipeline(Group(rank, 2))
                # Only the local layers are visible after pipeline().
                self.assertEqual(
                    [l.self_attn.layer_idx for l in model.layers], expected
                )
                self.assertIs(model.model, model.language_model.model)
                self.assertEqual(len(model.make_cache()), 3)

    def test_gemma4_shard(self):
        test_configs = {
            "plain": {},
            "kv_shared": GEMMA4_KV_SHARED,
            "k_eq_v": {"attention_k_eq_v": True, "num_global_key_value_heads": 2},
            "moe": GEMMA4_MOE,
        }
        for name, overrides in test_configs.items():
            with self.subTest(name):
                model = make_gemma4_text(**overrides)
                x = mx.random.randint(0, 128, shape=(2, 7))
                expected = model(x)
                if name == "moe":
                    gate = model.model.layers[0].experts.switch_glu.gate_proj
                    shape = gate.weight.shape
                model.shard()
                out = model(x)
                self.assertTrue(mx.allclose(expected, out, rtol=1e-3, atol=1e-3))
                if name == "moe":
                    for layer in model.model.layers:
                        self.assertIsNotNone(layer.experts.sharding_group)
                    gate = model.model.layers[0].experts.switch_glu.gate_proj
                    # A group of one keeps all experts and all columns.
                    self.assertEqual(gate.weight.shape, shape)

    def test_gemma4_wrapper_shard(self):
        config = {
            "model_type": "gemma4",
            "vocab_size": 128,
            "text_config": gemma4_text_config(**GEMMA4_MOE),
        }
        model = gemma4.Model(gemma4.ModelArgs.from_dict(config))
        x = mx.random.randint(0, 128, shape=(2, 7))
        expected = model(x)
        model.shard()
        self.assertTrue(mx.allclose(expected, model(x), rtol=1e-3, atol=1e-3))
        self.assertIsNotNone(model.layers[0].experts.sharding_group)

    def test_gemma4_shard_divisibility(self):
        # The base fields divide by 3, so each case fails on one field only.
        base = {
            "num_attention_heads": 6,
            "num_key_value_heads": 6,
            "intermediate_size": 96,
        }
        k_eq_v = {"attention_k_eq_v": True, "num_global_key_value_heads": 3}
        failing = [
            ("num_attention_heads", {"num_attention_heads": 4}),
            ("num_key_value_heads", {"num_key_value_heads": 2}),
            ("intermediate_size", {"intermediate_size": 128}),
            (
                "num_global_key_value_heads",
                {**k_eq_v, "num_global_key_value_heads": 2},
            ),
            ("moe_intermediate_size", {**GEMMA4_MOE, "moe_intermediate_size": 32}),
        ]
        for field, overrides in failing:
            with self.subTest(field=field):
                model = make_gemma4_text(**{**base, **overrides})
                with self.assertRaisesRegex(ValueError, rf"\b{field}="):
                    model.shard(Group(0, 3))

        # Fields that the model does not use are not checked.
        passing = [
            ("valid", {}),
            ("valid k_eq_v", k_eq_v),
            ("valid moe", {**GEMMA4_MOE, "moe_intermediate_size": 33}),
            ("moe off, size set", {"moe_intermediate_size": 32}),
            ("moe off, size None", {"moe_intermediate_size": None}),
            (
                "k_eq_v off",
                {"attention_k_eq_v": False, "num_global_key_value_heads": 2},
            ),
        ]
        for name, overrides in passing:
            with self.subTest(name):
                model = make_gemma4_text(**{**base, **overrides})
                model.shard(Group(0, 3))

    def test_pipeline_layer_assignment(self):
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
        with self.assertRaisesRegex(ValueError, "3 entries for group size 2"):
            assignment(4, 2, split=[1, 1, 2])
        with self.assertRaisesRegex(ValueError, "positive and sum to 4 layers"):
            assignment(4, 2, split=[0, 4])
        with self.assertRaisesRegex(ValueError, "positive and sum to 4 layers"):
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

        size = 2
        qwen2_config = {
            "model_type": "qwen2",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 7,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
        }
        qwen3_next_config = {
            "model_type": "qwen3_next",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 7,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 32,
            "mlp_only_layers": [0],
            "moe_intermediate_size": 32,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 256,
            "full_attention_interval": 4,
        }
        for arch, arch_config in (
            (qwen3_moe, config),
            (qwen2, qwen2_config),
            (qwen3_next, qwen3_next_config),
        ):
            with self.subTest(model_type=arch_config["model_type"]):
                args = arch.ModelArgs.from_dict(arch_config)
                assigned = []
                for rank in range(size):
                    model = arch.Model(args)
                    model.model.pipeline(Group(rank, size))
                    assigned.extend(
                        i for i, l in enumerate(model.model.layers) if l is not None
                    )
                self.assertEqual(sorted(assigned), list(range(7)))

        # Hybrid edge case: 7 layers on 2 ranks puts layers 0-2 (all linear,
        # full_attention_interval=4) on the last rank; its rescan must leave
        # fa_idx as None rather than pointing at a wrong layer. Rank 0 holds
        # layers 3-6, so its full-attention layer sits at local index 0.
        args = qwen3_next.ModelArgs.from_dict(qwen3_next_config)
        model = qwen3_next.Model(args)
        model.model.pipeline(Group(1, size))
        self.assertIsNone(model.model.fa_idx)
        self.assertEqual(model.model.ssm_idx, 0)
        model = qwen3_next.Model(args)
        model.model.pipeline(Group(0, size))
        self.assertEqual(model.model.fa_idx, 0)
        self.assertEqual(model.model.ssm_idx, 1)

        args = qwen3_next.ModelArgs.from_dict(
            {**qwen3_next_config, "full_attention_interval": 1}
        )
        for rank in range(size):
            model = qwen3_next.Model(args)
            model.model.pipeline(Group(rank, size))
            self.assertIsNone(model.model.ssm_idx)
            self.assertEqual(model.model.fa_idx, 0)


if __name__ == "__main__":
    unittest.main()
