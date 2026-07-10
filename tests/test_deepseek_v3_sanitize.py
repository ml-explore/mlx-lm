# Copyright © 2024 Apple Inc.
import unittest

import mlx.core as mx

from mlx_lm.models import deepseek_v3


class TestDeepseekV3Sanitize(unittest.TestCase):
    def _make_model(self, num_hidden_layers=4):
        # A small deepseek_v3-arch config. Keep every layer dense
        # (first_k_dense_replace >= num_hidden_layers) so we don't need a
        # routed-expert config, and shrink the dims so construction is cheap.
        args = deepseek_v3.ModelArgs(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=None,
            first_k_dense_replace=num_hidden_layers,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            max_position_embeddings=64,
        )
        return deepseek_v3.Model(args)

    def test_strips_mtp_layer_by_depth(self):
        num_hidden_layers = 4
        model = self._make_model(num_hidden_layers)

        weights = {
            "model.embed_tokens.weight": mx.zeros((1,)),
            "lm_head.weight": mx.zeros((1,)),
            # An unused precomputed rotary-freq tensor that must be dropped too.
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((1,)),
        }
        # In-range decoder layers 0..num_hidden_layers-1.
        for l in range(num_hidden_layers):
            weights[f"model.layers.{l}.input_layernorm.weight"] = mx.zeros((1,))
            weights[f"model.layers.{l}.mlp.gate_proj.weight"] = mx.zeros((1,))
        # The multi-token-prediction layer lives at index == num_hidden_layers.
        mtp = num_hidden_layers
        weights[f"model.layers.{mtp}.embed_tokens.weight"] = mx.zeros((1,))
        weights[f"model.layers.{mtp}.enorm.weight"] = mx.zeros((1,))
        weights[f"model.layers.{mtp}.shared_head.head.weight"] = mx.zeros((1,))

        out = model.sanitize(dict(weights))
        out_keys = set(out.keys())

        # MTP-layer keys are dropped.
        for k in weights:
            if k.startswith(f"model.layers.{mtp}."):
                self.assertNotIn(k, out_keys)
        # In-range decoder-layer keys are kept.
        for l in range(num_hidden_layers):
            self.assertIn(f"model.layers.{l}.input_layernorm.weight", out_keys)
            self.assertIn(f"model.layers.{l}.mlp.gate_proj.weight", out_keys)
        # Non-layer weights are kept; the rotary inv_freq is dropped.
        self.assertIn("model.embed_tokens.weight", out_keys)
        self.assertIn("lm_head.weight", out_keys)
        self.assertNotIn("model.layers.0.self_attn.rotary_emb.inv_freq", out_keys)


if __name__ == "__main__":
    unittest.main()
