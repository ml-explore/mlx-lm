# Copyright © 2026 Apple Inc.

import json
import re
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models import inkling2
from mlx_lm.processors import inkling2 as inkling2_processor
from mlx_lm.processors import load_processor
from mlx_lm.utils import load_model, quantize_model

CONFIG = {
    "model_type": "inkling2",
    "image_token_id": 127,
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
        "sliding_window_size": 8,
        "local_layer_ids": [0, 1, 2, 3, 4, 6],
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 3.0,
        "rel_l2_clip_threshold": 2.0,
        "sconv_kernel_size": 4,
        "n_layers_per_attn_res_block": 3,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 64,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 2.0,
        "final_logit_softcapping": 30.0,
    },
}

VISION_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "native_patch_size": 2,
    "adapter_merge_size": 2,
    "posemb_grid_size": 4,
    "projection_hidden_dim": 48,
}

UNPADDED_VOCAB = CONFIG["text_config"]["unpadded_vocab_size"]


def create_model(vision: bool = False) -> inkling2.Model:
    mx.random.seed(0)
    config = {**CONFIG, "vision_config": VISION_CONFIG} if vision else CONFIG
    model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale", "lambda_1", "lambda_2")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 16
            v = mx.random.normal(v.shape) * fan_in**-0.5
        weights.append((k, v))
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


def quantize_experts(model: inkling2.Model) -> None:
    def predicate(path, module):
        if ".mlp.experts." not in path:
            return False
        if ".layers.7." in path:
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}

    quantize_model(model, {}, None, None, quant_predicate=predicate)


_CHECKPOINT_NAMES = [
    (r"^lm_head\.", "model.llm.unembed."),
    (r"^model\.language_model\.embed_tokens\.", "model.llm.embed."),
    (r"^model\.language_model\.", "model.llm."),
    (r"\.final_input\.query$", ".attn_res.output_query"),
    (r"\.layers\.(\d+)\.mlp_input\.query$", r".attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.", ".attn.wv_dv."),
    (r"\.self_attn\.r_proj\.", ".attn.wr_du."),
    (r"\.self_attn\.o_proj\.", ".attn.wo_ud."),
    (r"\.self_attn\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.", ".attn."),
    (r"\.post_attention_convolution\.", ".attn_sconv."),
    (r"\.post_mlp_convolution\.", ".mlp_sconv."),
    (r"\.conv1d\.weight$", ".weight"),
    (r"sconv\.static_weight$", "sconv.static_w"),
    (r"sconv\.weight_proj\.weight$", "sconv.weight_proj"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
    (r"^model\.vision_tower\.pre_layernorm\.", "model.visual.backbone.ln_pre."),
    (r"^model\.vision_tower\.merger\.linear_1\.", "model.visual.proj_in."),
    (r"^model\.vision_tower\.merger\.linear_2\.", "model.visual.proj_out."),
    (r"^model\.vision_tower\.", "model.visual.backbone."),
    (r"position_embedding\.weight$", "position_embedding"),
    (r"\.layernorm_before\.", ".ln_1."),
    (r"\.layernorm_after\.", ".ln_2."),
    (r"\.mlp\.fc1\.", ".mlp.c_fc."),
    (r"\.mlp\.fc2\.", ".mlp.c_proj."),
    (r"\.lambda_(\d)$", r".ls_\1.gamma"),
]


def checkpoint(model: inkling2.Model) -> dict:
    weights = dict(tree_flatten(model.parameters()))
    ckpt = {}
    denominator = mx.array([7.5])
    for k, v in weights.items():
        if m := re.match(r"(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$", k):
            prefix, name, param = m.groups()
            prefix = prefix.replace("model.language_model.", "model.llm.")
            w = f"{prefix}.{'w13' if name == 'gate_up' else 'w2'}_weight"
            if param == "weight":
                ckpt[w] = v.view(mx.uint8)
            elif param == "scales":
                ckpt[f"{w}_scale"] = v
            else:
                ckpt[f"{w}_amax"] = v * denominator / 2688.0
                ckpt[f"{w}_outer_scale_denominator"] = denominator
            continue
        if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
            ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
            continue
        for pattern, replacement in _CHECKPOINT_NAMES:
            k = re.sub(pattern, replacement, k)
        if k.endswith("sconv.weight"):
            v = v.swapaxes(1, 2)
        ckpt[k] = v
    return ckpt


def image_prompt(model: inkling2.Model, *, h: int, w: int):
    vision = model.args.vision
    side = vision.native_patch_size * vision.adapter_merge_size
    rows = (h // side) * (w // side)
    image = mx.random.uniform(-1, 1, shape=(3, h, w))
    inputs = mx.array([1, 2] + [model.args.image_token_id] * rows + [3, 4])
    return image, inputs


class TestInkling2(unittest.TestCase):
    def test_forward(self):
        model = create_model()
        for dtype in (mx.float32, mx.bfloat16):
            model.update(
                tree_unflatten(
                    [(k, v.astype(dtype)) for k, v in tree_flatten(model.parameters())]
                )
            )
            outputs = model(mx.array([[0, 1]]))
            self.assertEqual(outputs.shape, (1, 2, UNPADDED_VOCAB))
            self.assertEqual(outputs.dtype, mx.float32)

    def test_cached_matches_forward(self):
        model = create_model()
        tokens = mx.random.randint(0, UNPADDED_VOCAB, (2, 20))
        full = model(tokens)

        cache = model.make_cache()
        steps = [model(tokens[:, :6], cache=cache), model(tokens[:, 6:12], cache=cache)]
        steps += [model(tokens[:, i : i + 1], cache=cache) for i in range(12, 20)]
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )

    def test_batch_matches_single(self):
        model = create_model()
        prompts = [[1, 2, 3], list(range(5, 30)), list(range(40, 52))]
        max_tokens = 10

        expected = []
        for p in prompts:
            steps = generate_step(mx.array(p), model, max_tokens=max_tokens)
            expected.append([(int(t), lp) for t, lp in steps])

        gen = BatchGenerator(model, max_tokens=max_tokens, prefill_step_size=7)
        uids = gen.insert(prompts)
        got = {uid: [] for uid in uids}
        while any(len(g) < max_tokens for g in got.values()):
            for r in gen.next_generated():
                got[r.uid].append((r.token, r.logprobs))
        gen.close()

        for uid, want in zip(uids, expected):
            self.assertEqual([t for t, _ in got[uid]], [t for t, _ in want])
            for (_, a), (_, b) in zip(got[uid], want):
                self.assertTrue(mx.allclose(a, b, atol=1e-4))

    def test_sanitize_checkpoint(self):
        model = create_model(vision=True)
        quantize_experts(model)
        weights = dict(tree_flatten(model.parameters()))
        sanitized = model.sanitize(checkpoint(model))
        self.assertEqual(sanitized.keys(), weights.keys())
        for k, v in weights.items():
            self.assertTrue(mx.allclose(sanitized[k], v), k)

        sanitized = model.sanitize(dict(weights))
        for k, v in weights.items():
            self.assertTrue(mx.array_equal(sanitized[k], v), k)

    def test_load_checkpoint(self):
        model = create_model(vision=True)
        quantize_experts(model)
        tokens = mx.array([[5, 6, 7, 8]])
        expected = model(tokens)

        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            mx.save_safetensors(str(path / "model.safetensors"), checkpoint(model))
            config = {**CONFIG, "vision_config": VISION_CONFIG}
            (path / "config.json").write_text(json.dumps(config))
            quant = {"quant_method": "inkling2"}
            (path / "hf_quant_config.json").write_text(json.dumps(quant))
            loaded, _ = load_model(path)
        self.assertTrue(mx.allclose(loaded(tokens), expected, rtol=1e-4, atol=1e-4))

    def test_embed_inputs(self):
        model = create_model(vision=True)
        image, inputs = image_prompt(model, h=8, w=12)
        rows = model.encode_image(image)
        self.assertEqual(rows.shape, (6, CONFIG["text_config"]["hidden_size"]))
        h = model.embed_inputs(inputs, pixel_values=[image])
        text = model.model.language_model.embed(inputs)
        self.assertTrue(mx.array_equal(h[2:8], rows))
        self.assertTrue(mx.array_equal(h[:2], text[:2]))
        self.assertTrue(mx.array_equal(h[8:], text[8:]))

        out = model(inputs[None], pixel_values=[image])
        self.assertTrue(mx.allclose(out, model(inputs[None], input_embeddings=h[None])))

    def test_processor(self):
        processor = inkling2_processor.Processor(block_size=4)
        image = np.full((5, 8, 3), 255, dtype=np.uint8)
        pixels, num_tokens = processor.preprocess_image(image)
        self.assertEqual(pixels.shape, (3, 8, 8))
        self.assertEqual(num_tokens, 4)
        self.assertTrue(mx.all(pixels[:, :5] == 1.0).item())
        self.assertTrue(mx.allclose(pixels[:, 5:], mx.array((-1 / 255 - 0.5) / 0.5)))

        token = inkling2_processor.IMAGE_TOKEN
        text, pixel_values = processor(f"A {token} B", images=[image])
        self.assertEqual(text, f"A {token * 4} B")
        self.assertEqual(len(pixel_values), 1)
        with self.assertRaisesRegex(ValueError, "for 2 images"):
            processor(f"A {token}", images=[image, image])

        config = {**CONFIG, "vision_config": VISION_CONFIG}
        processor = load_processor(create_model(vision=True), config)
        self.assertIsInstance(processor, inkling2_processor.Processor)
        self.assertEqual(processor.block_size, 4)


if __name__ == "__main__":
    unittest.main()
