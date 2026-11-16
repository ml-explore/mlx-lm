# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import ministral3, mistral3
from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS, awq_quantize

VOCAB_SIZE = 128


def ministral3_config():
    return {
        "model_type": "ministral3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 32,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.5,
            "original_max_position_embeddings": 512,
        },
    }


def llama_config():
    return {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
    }


class TestAWQ(unittest.TestCase):
    def test_ministral3_attn_scale_default(self):
        # AWQ calls the blocks without attn_scale, so the fallback must match
        # what the model passes down. Otherwise AWQ calibrates the wrong graph.
        model = ministral3.Model(ministral3.ModelArgs.from_dict(ministral3_config()))
        attn = model.model.layers[0].self_attn
        x = mx.random.normal((1, 5, 64))
        expected = ministral3._get_llama_4_attn_scale(5, 0, 0.5, 512).astype(x.dtype)
        self.assertTrue(mx.allclose(attn(x), attn(x, expected), atol=1e-6).item())

    def test_ministral3_awq_quantize(self):
        config = ministral3_config()
        model = ministral3.Model(ministral3.ModelArgs.from_dict(config))
        self._quantize(model)
        self._assert_quantized(model, prefix="")

    def test_mistral3_awq_quantize(self):
        # The wrapper needs lm_key to descend into the language model, and it
        # builds either a llama or a ministral3 depending on text_config.
        for model_type in ["mistral3", "llava"]:
            for text_config in [llama_config(), ministral3_config()]:
                inner = text_config["model_type"]
                with self.subTest(model_type=model_type, language_model=inner):
                    model = mistral3.Model(
                        mistral3.ModelArgs(
                            model_type=model_type, text_config=text_config
                        )
                    )
                    self._quantize(model)
                    self._assert_quantized(model, prefix="language_model.")

    def _quantize(self, model):
        awq_quantize(
            model,
            mx.random.randint(0, VOCAB_SIZE, (2, 16)),
            AWQ_MODEL_CONFIGS[model.model_type],
            group_size=32,
            bits=4,
            embed_group_size=32,
        )
        mx.eval(model.parameters())

    def _assert_quantized(self, model, prefix):
        keys = {k for k, _ in tree_flatten(model.parameters())}
        for key in [
            "model.embed_tokens.scales",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.mlp.gate_proj.scales",
            "lm_head.scales",
        ]:
            self.assertIn(prefix + key, keys)


if __name__ == "__main__":
    unittest.main()
