# Copyright © 2026 Apple Inc.

import unittest
from types import SimpleNamespace

import mlx.core as mx
import test_models
from mlx.utils import tree_flatten, tree_map
from model_parallel_tests import Group

from mlx_lm.chat_templates.muse_spark_open import RESPONSE_FORMAT
from mlx_lm.generate import TextStateMachine, make_text_state_machine
from mlx_lm.models import muse_spark_open
from mlx_lm.models.cache import RotatingKVCache

# Layers 0-2 are sliding (window 8) with RoPE, layer 3 is full and NoPE.
CONFIG = {
    "model_type": "muse_spark_open",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "rms_post_norm_eps": 1e-8,
    "rope_theta": 10000.0,
    "every_n_layers_nope": 4,
    "global_attn_cfg": [8, 8, 8, 0],
    "output_soft_cap_temp": 20.0,
    "use_fp32_for_output": True,
    # The default id is not in this vocab.
    "patch_token_id": 255,
    "moe": {
        "is_enabled": True,
        "num_experts": 4,
        "topk": 2,
        "compression_factor": 2,
        "override_routed_hidden_dim": 32,
        "router_in_fp32": True,
        "router_score_division_eps": 1e-15,
        "route_scale": 1.5,
    },
    "metap": {
        "use_metap": True,
        "base_width": 32,
        "router_has_one_inf_dim": True,
        "m_emb": 2.0,
    },
}

# Tiles of 4 x 4 patches; layer 0 attends in tiles, layer 1 to all patches.
VISION_CONFIG = {
    "navit_config": {
        "vision_output_dim": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "image_size": 8,
        "patch_size": 2,
        "sparse_attention_factor": 2,
    },
    "projector_input_dim": 48,
}


def create_model(vision: bool = False) -> muse_spark_open.Model:
    mx.random.seed(0)
    config = {**CONFIG, "vision_config": VISION_CONFIG} if vision else CONFIG
    args = muse_spark_open.ModelArgs.from_dict(config)
    return muse_spark_open.Model(args)


def image_prompt(
    model: muse_spark_open.Model, *, h: int, w: int
) -> tuple[mx.array, mx.array]:
    """An h x w image and a prompt with a patch token for each of its rows."""
    vision = model.args.vision
    side = vision.patch_size * vision.downsample_factor
    rows = (h // side) * (w // side)
    image = mx.random.uniform(shape=(3, h, w))
    inputs = mx.array([1, 2] + [model.args.patch_token_id] * rows + [3, 4])
    return image, inputs


class TestMuseSparkOpen(unittest.TestCase):
    def test_forward(self):
        model = create_model()
        for t in [mx.float32, mx.bfloat16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))
            # The soft-capped logits are fp32 for all weight dtypes.
            outputs = model(mx.array([[0, 1]]))
            self.assertEqual(outputs.shape, (1, 2, model.args.vocab_size))
            self.assertEqual(outputs.dtype, mx.float32)

    def test_left_padding(self):
        # Its prompts fit in the sliding window.
        test_models.TestModels.left_padding_test_runner(self, create_model())

    def test_cached_matches_forward(self):
        model = create_model()
        tokens = mx.random.randint(0, model.args.vocab_size, (2, 16))
        full = model(tokens)

        # Prefill in two chunks, then decode past the sliding window.
        cache = model.make_cache()
        steps = [model(tokens[:, :6], cache=cache), model(tokens[:, 6:12], cache=cache)]
        steps += [model(tokens[:, i : i + 1], cache=cache) for i in range(12, 16)]
        # The last hidden state is bf16.
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-3, atol=1e-3)
        )

    def test_sanitize_converted_weights(self):
        model = create_model(vision=True)
        weights = dict(tree_flatten(model.parameters()))
        sanitized = model.sanitize(dict(weights))
        # The norms must not get the +1 again.
        self.assertEqual(sanitized.keys(), weights.keys())
        for k, v in weights.items():
            self.assertTrue(mx.array_equal(sanitized[k], v), k)

        # Checkpoint norms are centered at zero.
        norm = "model.layers.0.pre_ffn_norm.weight"
        raw = {norm: mx.zeros((64,)), "lm_head.weight": weights["lm_head.weight"]}
        self.assertTrue(mx.array_equal(model.sanitize(raw)[norm], mx.ones((64,))))

    def test_sanitize_muse_spark_weights(self):
        # Muse Spark moves the vision prefix and adds an audio tower.
        vision = "modalities.vision_perception.encoder."
        names = [
            "lm_head.weight",
            vision + "vision_projection.weight",
            vision + "vision_model.ln_pre.weight",
            vision + "vision_module.wqkv.bias",
            "modalities.soft_audio_perception.interaction_encoder.norm.weight",
        ]
        model = create_model(vision=True)
        sanitized = model.sanitize({k: mx.zeros((1,)) for k in names})
        self.assertEqual(
            sanitized.keys(), {"lm_head.weight", "vision_projection.weight"}
        )

    def test_pipeline_layer_indices(self):
        # Rank 1 has layers 0-1 (sliding), rank 0 has 2 (sliding) and 3 (full).
        for rank, fa_idx, sliding in ((0, 1, [True, False]), (1, 0, [True, True])):
            model = create_model()
            model.model.pipeline(Group(rank, 2))
            self.assertEqual(model.model.fa_idx, fa_idx)
            self.assertEqual(model.model.swa_idx, 0)
            caches = model.make_cache()
            self.assertEqual([isinstance(c, RotatingKVCache) for c in caches], sliding)

    def test_embed_inputs(self):
        model = create_model(vision=True)
        # 6 x 8 patches, more than one tile, give 3 x 4 rows.
        image, inputs = image_prompt(model, h=12, w=16)
        rows = model.encode_image(image)
        self.assertEqual(rows.shape, (12, model.args.hidden_size))
        h = model.embed_inputs(inputs, pixel_values=[image])
        text = model.model.embed(inputs)
        self.assertTrue(mx.array_equal(h[2:14], rows))
        self.assertTrue(mx.array_equal(h[:2], text[:2]))
        self.assertTrue(mx.array_equal(h[14:], text[14:]))

    def test_prefill_input_embeddings(self):
        model = create_model(vision=True)
        image, inputs = image_prompt(model, h=12, w=16)
        h = model.embed_inputs(inputs, pixel_values=[image])[None]
        inputs = inputs[None]
        one = model(inputs, cache=model.make_cache(), input_embeddings=h)

        # The first chunk ends in the image rows.
        cache = model.make_cache()
        steps = [
            model(inputs[:, :7], cache=cache, input_embeddings=h[:, :7]),
            model(inputs[:, 7:], cache=cache, input_embeddings=h[:, 7:]),
        ]
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), one, rtol=1e-3, atol=1e-3)
        )


class TestMuseSparkOpenResponseFormat(unittest.TestCase):
    def split(self, text, step):
        tokenizer = SimpleNamespace(
            has_thinking=False,
            has_tool_calling=True,
            tool_call_start="<atem:function_calls>",
            tool_call_end="</atem:function_calls>",
            structural_markers=(),
            response_transitions=RESPONSE_FORMAT["transitions"],
        )
        sm = make_text_state_machine(tokenizer)
        state = sm.make_state(RESPONSE_FORMAT["initial_state"])
        out = {"reasoning": "", "tool": "", "normal": ""}
        for i in range(0, len(text), step):
            state, segments = TextStateMachine.split(state, text[i : i + step])
            for name, chunk in segments:
                if name in out:
                    out[name] += chunk
        return out

    def test_message_headers(self):
        call = (
            '<atem:function_calls><atem:invoke name="grep">'
            '<atem:parameter name="pattern">a</atem:parameter>'
            "</atem:invoke></atem:function_calls>"
        )
        for start in ("<|start|>", ""):
            text = (
                " to=self<|message|>think<|eom|>"
                f"{start}assistant to=commentary<|message|>one<|eom|>"
                f"{start}assistant to=grep<|message|>{call}<|eom|>"
                f"{start}assistant to=user<|message|>two"
            )
            for step in (1, 3, 7, len(text)):
                out = self.split(text, step)
                self.assertEqual(out["reasoning"], "think")
                self.assertEqual(out["normal"], "onetwo")
                self.assertEqual(
                    out["tool"],
                    call[len("<atem:function_calls>") : -len("</atem:function_calls>")],
                )


if __name__ == "__main__":
    unittest.main()
