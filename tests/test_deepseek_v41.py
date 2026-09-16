"""Small Transformers-to-MLX parity checks for DeepSeek-V4.1."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm import utils
from mlx_lm.models.cache import make_prompt_cache


HAS_TRANSFORMERS_V41 = importlib.util.find_spec("transformers.models.deepseek_v41") is not None


def _tiny_config():
    from transformers import DeepseekV41TextConfig

    return DeepseekV41TextConfig(
        vocab_size=128,
        hidden_size=72,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=18,
        q_lora_rank=18,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        sliding_window=8,
        compress_ratios=[0, 2, 1, 1],
        kv_source_layer_ids=[1, 2],
        index_source_layer_ids=[1, 2, 3],
        index_n_heads=2,
        index_head_dim=6,
        index_topk=4,
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        num_nextn_predict_layers=0,
        max_position_embeddings=32,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 2.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 16,
        },
        rms_norm_eps=1e-5,
        dspark_noise_token_id=127,
    )


@unittest.skipUnless(HAS_TRANSFORMERS_V41, "requires the pending Transformers DeepSeek-V4.1 implementation")
class DeepseekV41Test(unittest.TestCase):
    def test_tiny_transformers_checkpoint_loads_and_matches(self):
        import torch
        from transformers import DeepseekV41Config, DeepseekV41ForCausalLM

        torch.manual_seed(7)
        text_config = _tiny_config()
        hf_model = DeepseekV41ForCausalLM(DeepseekV41Config(text_config=text_config))
        hf_model.eval()
        input_ids = torch.tensor([[3, 7, 11, 19, 23]], dtype=torch.long)

        with tempfile.TemporaryDirectory() as directory:
            hf_model.save_pretrained(directory, safe_serialization=True)
            mlx_model, _ = utils.load_model(Path(directory))

            with torch.no_grad():
                expected = hf_model(input_ids).logits.detach().cpu().numpy()
            actual = mlx_model(mx.array(input_ids.numpy(), dtype=mx.uint32))
            mx.eval(actual)

            self.assertLess(sum(p.numel() for p in hf_model.parameters()), 100_000_000)
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-4, atol=2e-4)

    def test_cached_decode_matches_full_forward(self):
        import torch
        from transformers import DeepseekV41Config, DeepseekV41ForCausalLM

        torch.manual_seed(9)
        hf_model = DeepseekV41ForCausalLM(DeepseekV41Config(text_config=_tiny_config()))
        hf_model.eval()
        with tempfile.TemporaryDirectory() as directory:
            hf_model.save_pretrained(directory, safe_serialization=True)
            mlx_model, _ = utils.load_model(Path(directory))

            ids = mx.array([[3, 7, 11, 19, 23]], dtype=mx.uint32)
            full = mlx_model(ids)
            cache = make_prompt_cache(mlx_model)
            mlx_model(ids[:, :-1], cache=cache)
            decoded = mlx_model(ids[:, -1:], cache=cache)
            mx.eval(full, decoded)
            np.testing.assert_allclose(
                np.asarray(decoded), np.asarray(full[:, -1:]), rtol=2e-4, atol=2e-4
            )


if __name__ == "__main__":
    unittest.main()
