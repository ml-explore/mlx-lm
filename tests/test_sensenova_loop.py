import unittest

import mlx.core as mx

from mlx_lm.models.sensenova_loop import Model, ModelArgs, TextModel


def _text_config(num_hidden_layers=8):
    return dict(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-5,
        vocab_size=128,
        tie_word_embeddings=False,
        full_attention_interval=4,
    )


LOOP = {"start": 2, "end": 6, "eval_count": 2, "counts": [2]}
MHC = {"start": 2, "end": 6, "streams": 2, "iterations": 20, "fixed_identity": False}


class TestSenseNovaLoop(unittest.TestCase):
    def test_execution_plan_reuses_shared_layers(self):
        tm = TextModel(_text_config(), loop_settings=LOOP, mhc_settings=MHC)
        plan = tm._build_execution_plan()
        # 2 before + 4 looped x2 + 2 after = 12 executions of 8 physical layers.
        self.assertEqual(len(plan), 12)
        self.assertEqual(
            [p for p, _ in plan], [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
        )
        # Every physical layer is the SAME module object across its visits.
        self.assertEqual(len({id(tm.model.layers[p]) for p, _ in plan}), 8)

    def test_mhc_wraps_only_the_loop_range(self):
        tm = TextModel(_text_config(), loop_settings=LOOP, mhc_settings=MHC)
        from mlx_lm.models.sensenova_loop import MHCDecoderLayer

        self.assertFalse(isinstance(tm.model.layers[1], MHCDecoderLayer))
        for i in range(2, 6):
            self.assertTrue(isinstance(tm.model.layers[i], MHCDecoderLayer))
        self.assertFalse(isinstance(tm.model.layers[6], MHCDecoderLayer))

    def test_sanitize_drops_unmatched_towers_and_vestigial_parameter(self):
        model = Model(
            ModelArgs(
                model_type="sensenova_loop",
                text_config=_text_config(),
                loop_settings=LOOP,
                mhc=MHC,
            )
        )
        w = {
            "model.language_model.embed_tokens.weight": mx.zeros((128, 64)),
            "model.language_model.layers.0.mlp.gate_proj.weight": mx.zeros((128, 64)),
            "mtp.layers.0.weight": mx.zeros((64, 64)),
            "model.language_model.loop_gate.logits": mx.zeros((4,)),
            "model.visual.patch_embed.proj.weight": mx.zeros((64, 3, 4, 4)),
        }
        out = model.sanitize(w)
        keys = set(out)
        self.assertIn("language_model.model.embed_tokens.weight", keys)
        self.assertNotIn("mtp.layers.0.weight", keys)
        self.assertNotIn("language_model.model.loop_gate.logits", keys)
        self.assertNotIn("model.visual.patch_embed.proj.weight", keys)


if __name__ == "__main__":
    unittest.main()
