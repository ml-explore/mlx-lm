# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models import sarvam_mla
from mlx_lm.models.deepseek_v3.layers import DeepseekV3MLP, DeepseekV3MoE
from mlx_lm.utils import model_package


class TestSarvamMLA(unittest.TestCase):
    def small_args(self):
        return sarvam_mla.ModelArgs(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=4,
            kv_lora_rank=16,
            q_lora_rank=None,
            qk_rope_head_dim=8,
            v_head_dim=8,
            qk_nope_head_dim=8,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            max_position_embeddings=128,
            rope_parameters={
                "factor": 2.0,
                "original_max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rope_type": "deepseek_yarn",
            },
        )

    def test_config_mapping(self):
        args = sarvam_mla.ModelArgs.from_dict(
            {
                "model_type": "sarvam_mla",
                "num_experts": 128,
                "num_shared_experts": 1,
                "num_experts_per_tok": 8,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "rope_parameters": {
                    "factor": 40.0,
                    "rope_theta": 10000.0,
                    "rope_type": "deepseek_yarn",
                },
            }
        )
        self.assertEqual(args.n_routed_experts, 128)
        self.assertEqual(args.n_shared_experts, 1)
        self.assertEqual(args.n_group, 1)
        self.assertEqual(args.topk_group, 1)
        self.assertIsNone(args.q_lora_rank)
        self.assertEqual(args.rope_scaling["factor"], 40.0)
        self.assertEqual(args.rope_theta, 10000.0)
        self.assertEqual(model_package({"model_type": "sarvam_mla"}), "sarvam_mla")

    def test_model_structure_and_forward(self):
        args = self.small_args()
        model = sarvam_mla.Model(args)
        self.assertIsInstance(model.layers[0].mlp, DeepseekV3MLP)
        self.assertIsInstance(model.layers[1].mlp, DeepseekV3MoE)
        self.assertTrue(hasattr(model.layers[0].self_attn, "q_proj"))
        self.assertFalse(hasattr(model.layers[0].self_attn, "q_a_proj"))

        logits = model(mx.array([[1, 2, 3]]))
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, args.vocab_size))

    def test_checkpoint_conversion(self):
        args = self.small_args()
        model = sarvam_mla.Model(args)
        attn = "model.layers.1.self_attn"
        mlp = "model.layers.1.mlp"
        head_dim = args.qk_nope_head_dim + args.v_head_dim
        weights = {
            f"{attn}.kv_b_proj.weight": mx.arange(
                args.num_attention_heads * head_dim * args.kv_lora_rank
            ).reshape(args.num_attention_heads * head_dim, args.kv_lora_rank),
            f"{mlp}.gate.e_score_correction_bias": mx.array([1.0, 2.0, 3.0, 4.0]),
        }
        for expert in range(args.n_routed_experts):
            for projection, shape in (
                ("gate_proj", (args.moe_intermediate_size, args.hidden_size)),
                ("up_proj", (args.moe_intermediate_size, args.hidden_size)),
                ("down_proj", (args.hidden_size, args.moe_intermediate_size)),
            ):
                weights[f"{mlp}.experts.{expert}.{projection}.weight"] = mx.full(
                    shape, expert
                )

        converted = model.sanitize(weights)
        self.assertNotIn(f"{attn}.kv_b_proj.weight", converted)
        self.assertEqual(
            converted[f"{attn}.embed_q.weight"].shape,
            (args.num_attention_heads, args.kv_lora_rank, args.qk_nope_head_dim),
        )
        self.assertEqual(
            converted[f"{attn}.unembed_out.weight"].shape,
            (args.num_attention_heads, args.v_head_dim, args.kv_lora_rank),
        )
        self.assertEqual(
            converted[f"{mlp}.switch_mlp.gate_proj.weight"].shape,
            (
                args.n_routed_experts,
                args.moe_intermediate_size,
                args.hidden_size,
            ),
        )
        bias = converted[f"{mlp}.gate.e_score_correction_bias"]
        self.assertAlmostEqual(float(bias.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
