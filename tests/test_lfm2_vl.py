# Copyright © 2026 Apple Inc.

"""LFM2-VL image inputs: vision tower, projector, preprocessing, loading."""

import importlib
import io
import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from mlx_lm.models.lfm2_vl_vision import Projector, _resize_matrix
from mlx_lm.multimodal import Lfm2VlImageInputs

lfm2_vl = importlib.import_module("mlx_lm.models.lfm2-vl")

TEXT = {
    "model_type": "lfm2",
    "vocab_size": 300,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": 32,
    "block_multiple_of": 16,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": False,
    "intermediate_size": 64,
    "layer_types": ["conv", "full_attention"],
}
VISION = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "patch_size": 4,
    "num_patches": 16,
}


def _model():
    args = lfm2_vl.ModelArgs.from_dict(
        {
            "model_type": "lfm2_vl",
            "text_config": dict(TEXT),
            "vision_config": VISION,
            "image_token_id": 7,
            "projector_hidden_size": 24,
            "projector_use_layernorm": False,
        }
    )
    return lfm2_vl.Model(args)


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, format="PNG")
    return buf.getvalue()


class TestResize(unittest.TestCase):
    def test_same_size_is_identity(self):
        self.assertTrue(np.allclose(_resize_matrix(16, 16), np.eye(16)))

    def test_rows_sum_to_one(self):
        for n_out in (5, 16, 32):
            self.assertTrue(np.allclose(_resize_matrix(16, n_out).sum(1), 1))

    def test_downscale_by_two_averages_pairs(self):
        # Antialiased bilinear at scale 2: weights 1/8, 3/8, 3/8, 1/8.
        row = _resize_matrix(8, 4)[1]
        self.assertTrue(np.allclose(row[1:5], [1 / 8, 3 / 8, 3 / 8, 1 / 8]))


class TestProjector(unittest.TestCase):
    def test_unshuffle_matches_hf(self):
        # HF pixel_unshuffle, written with numpy on the same input.
        x = np.arange(1 * 4 * 6 * 3, dtype=np.float32).reshape(1, 4, 6, 3)
        f = 2
        ref = x.reshape(1, 4, 3, 6).transpose(0, 2, 1, 3)
        ref = ref.reshape(1, 3, 2, 12).transpose(0, 2, 1, 3)
        proj = Projector(3, 8, 8, f, True, "gelu", False)
        self.assertTrue(np.array_equal(np.array(proj.unshuffle(mx.array(x))), ref))


class TestModel(unittest.TestCase):
    def test_sanitize_hf_names(self):
        model = _model()
        params = dict(tree_flatten(model.parameters()))
        hf = {}
        for k, v in params.items():
            if k.startswith("language_model.model."):
                hf["model.language_model." + k[len("language_model.model.") :]] = v
            elif k.startswith("vision_tower."):
                hf["model.vision_tower.vision_model." + k[len("vision_tower.") :]] = v
            else:
                hf["model." + k] = v
        hf["lm_head.weight"] = mx.zeros((1,))
        out = model.sanitize(hf)
        self.assertEqual(set(out), set(params))

    def test_sanitize_keeps_mlx_names(self):
        model = _model()
        params = dict(tree_flatten(model.parameters()))
        self.assertEqual(set(model.sanitize(params)), set(params))

    def test_text_only_drops_vision(self):
        args = lfm2_vl.ModelArgs.from_dict(
            {"model_type": "lfm2_vl", "text_config": dict(TEXT)}
        )
        model = lfm2_vl.Model(args)
        self.assertIsNone(model.vision_tower)
        out = model.sanitize({"model.vision_tower.vision_model.x": mx.zeros(1)})
        self.assertEqual(out, {})

    def test_older_config_names(self):
        # model_type "lfm2-vl": image_token_index, vision_feature_layer -2
        # (the layers up to it stored), projector_use_layernorm unset.
        args = lfm2_vl.ModelArgs.from_dict(
            {
                "model_type": "lfm2-vl",
                "text_config": dict(TEXT),
                "vision_config": dict(VISION, num_hidden_layers=3),
                "image_token_index": 7,
                "vision_feature_layer": -2,
                "projector_hidden_size": 24,
                "projector_use_layernorm": None,
            }
        )
        model = lfm2_vl.Model(args)
        self.assertEqual(args.image_token_id, 7)
        self.assertEqual(len(model.vision_tower.encoder.layers), 2)
        self.assertIn("layer_norm", model.multi_modal_projector)

    def test_embed_with_images(self):
        model = _model()
        # Two 4x4-patch crops (2x2 tokens each) and one 2x4 (1x2 tokens).
        crops = [
            (mx.random.normal((16, 48)), 4, 4),
            (mx.random.normal((8, 48)), 2, 4),
            (mx.random.normal((16, 48)), 4, 4),
        ]
        ids = mx.array([[1] + [7] * 10 + [2]])
        embeds = model.embed_with_images(ids, crops)
        self.assertEqual(embeds.shape, (1, 12, 32))
        # The second crop's tokens are rows 5, 6: its own features.
        alone = model.image_features([crops[1]])
        self.assertTrue(mx.allclose(embeds[0, 5:7], alone, atol=1e-5).item())


class _FakeModel:
    class args:
        image_token_id = 7

    def embed_with_images(self, ids, crops):
        self.seen = (ids, crops)
        return mx.zeros((1, ids.shape[1], 8))


class TestImageInputs(unittest.TestCase):
    def setUp(self):
        ids = {"<|image_start|>": 100, "<|image_end|>": 101, "<|img_thumbnail|>": 102}
        for r in range(10):
            for c in range(10):
                ids[f"<|img_row_{r + 1}_col_{c + 1}|>"] = 200 + 10 * r + c
        self.model = _FakeModel()
        cfg = {"image_processor": {"image_mean": [0.5] * 3, "image_std": [0.5] * 3}}
        self.inputs = Lfm2VlImageInputs(self.model, cfg, ids)

    def test_small_image_is_one_crop(self):
        ids, _, key = self.inputs.build([1, 7, 2], [_png(320, 240)])
        # 320x240 -> 320x256 (x32), 10x8 tokens after the 2x2 unshuffle.
        self.assertEqual(ids, [1, 100] + [7] * 80 + [101, 2])
        self.assertEqual(len(key), len(ids))
        (_, h, w), = [(c[0], c[1], c[2]) for c in self.model.seen[1]]
        self.assertEqual((h, w), (16, 20))

    def test_large_image_is_tiled_with_thumbnail(self):
        ids, _, _ = self.inputs.build([7], [_png(2048, 1024)])
        crops = self.model.seen[1]
        # 2:1 -> a 2x1 grid... of 512 tiles is too few; the closest ratio with
        # the lowest tile count wins: (2, 1).
        tiles = [c for c in crops if (c[1], c[2]) == (32, 32)]
        self.assertEqual(len(tiles) + 1, len(crops))
        self.assertEqual(ids[0], 100)
        self.assertEqual(ids[1], 200)  # row 1 col 1
        self.assertIn(102, ids)  # thumbnail
        self.assertEqual(ids.count(7), 256 * len(tiles) + (crops[-1][1] // 2) * (crops[-1][2] // 2))

    def test_preprocessor_config_overrides(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_lm.multimodal import load_image_inputs

        model = _model()
        tokens = ["<|image_start|>", "<|image_end|>", "<|img_thumbnail|>", "<|img_row_1_col_1|>"]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "processor_config.json").write_text(json.dumps({"image_processor": {"max_tiles": 10}}))
            Path(d, "preprocessor_config.json").write_text(json.dumps({"max_tiles": 2, "resample": 2}))
            # Without the image tokens: no image inputs, not a KeyError later.
            self.assertIsNone(load_image_inputs(model, d))
            added = [{"content": t, "id": 100 + i} for i, t in enumerate(tokens)]
            Path(d, "tokenizer.json").write_text(json.dumps({"added_tokens": added}))
            inputs = load_image_inputs(model, d)
        self.assertEqual(inputs.max_tiles, 2)
        self.assertEqual(inputs.resample, 2, "older checkpoints resize bilinear")

    def test_placeholder_count_checked(self):
        with self.assertRaises(ValueError):
            self.inputs.build([1, 2], [_png(64, 64)])


if __name__ == "__main__":
    unittest.main()
