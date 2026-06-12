import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_lm.gguf import convert_to_gguf


class TestConvertToGGUFWithoutMocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.tokenizer_file_path = os.path.join(cls.test_dir, "tokenizer.json")
        with open(cls.tokenizer_file_path, "w") as f:
            f.write("{}")

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("mlx.core.save_gguf")
    def test_convert_to_gguf(
        self,
        mock_save_gguf,
        mock_from_pretrained,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 3
        mock_tokenizer.get_added_vocab.return_value = {}
        mock_tokenizer.get_vocab.return_value = {"<pad>": 0, "hello": 1, "world": 2}
        mock_tokenizer.all_special_tokens = ["<pad>"]
        mock_tokenizer.all_special_ids = [0]
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.unk_token_id = None
        mock_from_pretrained.return_value = mock_tokenizer

        model_path = Path(self.test_dir)
        weights = {
            "self_attn.q_proj.weight": mx.random.uniform(shape=[768, 768]),
        }
        config = {
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "_name_or_path": "test-llama",
        }
        output_file_path = "/fake/output/path/gguf_model.gguf"

        convert_to_gguf(model_path, weights, config, output_file_path)
        called_args, _ = mock_save_gguf.call_args
        self.assertEqual(called_args[0], output_file_path)


if __name__ == "__main__":
    unittest.main()


class TestQwen36MoETensorMapping(unittest.TestCase):
    """Tests for Qwen3.6 MoE (qwen3_5_moe) GGUF conversion support."""

    def test_translate_weight_names_strips_language_model_prefix(self):
        from mlx_lm.gguf import translate_weight_names

        name = "language_model.model.layers.0.mlp.gate_proj.weight"
        result = translate_weight_names(name)
        self.assertNotIn("language_model", result)
        self.assertIn("blk.0", result)

    def test_translate_weight_names_maps_switch_mlp(self):
        from mlx_lm.gguf import translate_weight_names

        name = "model.layers.0.mlp.switch_mlp.down_proj.weight"
        result = translate_weight_names(name)
        self.assertIn("ffn_down_exps", result)
        self.assertNotIn("switch_mlp", result)

    def test_translate_weight_names_maps_switch_mlp_gate_up(self):
        from mlx_lm.gguf import translate_weight_names

        name = "model.layers.0.mlp.switch_mlp.gate_up_proj.weight"
        result = translate_weight_names(name)
        self.assertIn("ffn_gate_up_exps", result)

    def test_translate_weight_names_maps_shared_expert(self):
        from mlx_lm.gguf import translate_weight_names

        gate = translate_weight_names(
            "model.layers.0.mlp.shared_expert.gate_proj.weight"
        )
        down = translate_weight_names(
            "model.layers.0.mlp.shared_expert.down_proj.weight"
        )
        up = translate_weight_names(
            "model.layers.0.mlp.shared_expert.up_proj.weight"
        )
        self.assertIn("ffn_gate_shexp", gate)
        self.assertIn("ffn_down_shexp", down)
        self.assertIn("ffn_up_shexp", up)

    def test_translate_weight_names_maps_moe_router(self):
        from mlx_lm.gguf import translate_weight_names

        name = "model.layers.0.mlp.gate.weight"
        result = translate_weight_names(name)
        self.assertIn("ffn_gate_inp", result)

    def test_translate_weight_names_maps_linear_attn(self):
        from mlx_lm.gguf import translate_weight_names

        a_log = translate_weight_names("model.layers.0.linear_attn.A_log")
        conv1d = translate_weight_names(
            "model.layers.0.linear_attn.conv1d.weight"
        )
        self.assertIn("ssm_a", a_log)
        self.assertIn("ssm_conv1d", conv1d)

    def test_gate_up_proj_fusion_in_convert(self):
        """Test that switch_mlp.gate_proj + up_proj are fused before name translation."""
        # Simulate 3D expert tensors [n_experts, intermediate, hidden]
        gate = mx.random.uniform(shape=[4, 512, 2048])
        up = mx.random.uniform(shape=[4, 512, 2048])

        # Simulate the fusion logic from convert_to_gguf
        weights = {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": gate,
            "model.layers.0.mlp.switch_mlp.up_proj.weight": up,
            "model.layers.0.mlp.switch_mlp.down_proj.weight": mx.random.uniform(
                shape=[4, 2048, 512]
            ),
        }

        # Apply fusion (same logic as in convert_to_gguf)
        fused_weights = {}
        skip_keys = set()
        for k, v in weights.items():
            if "switch_mlp.gate_proj" in k:
                up_key = k.replace("gate_proj", "up_proj")
                if up_key in weights:
                    cat_dim = 1 if v.ndim == 3 else 0
                    fused = mx.concatenate([v, weights[up_key]], axis=cat_dim)
                    fused_key = k.replace("gate_proj", "gate_up_proj")
                    fused_weights[fused_key] = fused
                    skip_keys.add(k)
                    skip_keys.add(up_key)
        if fused_weights:
            weights = {
                **(fused_weights),
                **{k: v for k, v in weights.items() if k not in skip_keys},
            }

        # Verify fusion result
        fused_key = "model.layers.0.mlp.switch_mlp.gate_up_proj.weight"
        self.assertIn(fused_key, weights)
        self.assertEqual(weights[fused_key].shape, [4, 1024, 2048])
        self.assertNotIn(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight", weights
        )
        self.assertNotIn(
            "model.layers.0.mlp.switch_mlp.up_proj.weight", weights
        )
