# Copyright © 2025 Apple Inc.

import unittest

from mlx_lm.models import mistral3


def _minimal_args():
    text_config = {
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
    }
    return mistral3.ModelArgs(model_type="mistral3", text_config=text_config)


class TestMistral3Sanitize(unittest.TestCase):
    def test_transformers_v5_nested_layout(self):
        model = mistral3.Model(_minimal_args())

        # transformers >= 5 nested "model." layout: the text backbone lives
        # directly under "model.language_model.", lm_head at the root.
        weights = {
            "model.language_model.embed_tokens.weight": 0,
            "model.language_model.layers.0.self_attn.q_proj.weight": 1,
            "model.language_model.layers.0.mlp.gate_proj.weight": 2,
            "lm_head.weight": 3,
            "model.vision_tower.foo": 4,
            "model.multi_modal_projector.bar": 5,
        }

        out = model.sanitize(weights)

        # language_model.* remapped, vision/projector dropped.
        self.assertIn("language_model.model.embed_tokens.weight", out)
        self.assertIn("language_model.model.layers.0.self_attn.q_proj.weight", out)
        self.assertIn("language_model.model.layers.0.mlp.gate_proj.weight", out)
        self.assertIn("language_model.lm_head.weight", out)

        for k in out:
            self.assertFalse(k.startswith("model."))
            self.assertNotIn("vision_tower", k)
            self.assertNotIn("multi_modal_projector", k)

    def test_nested_branch_triggers(self):
        # A key in the nested layout should route through the remap branch.
        model = mistral3.Model(_minimal_args())
        weights = {
            "model.language_model.embed_tokens.weight": 0,
            "model.vision_tower.foo": 1,
        }
        out = model.sanitize(weights)
        self.assertIn("language_model.model.embed_tokens.weight", out)
        self.assertTrue(all(not k.startswith("model.") for k in out))


if __name__ == "__main__":
    unittest.main()
