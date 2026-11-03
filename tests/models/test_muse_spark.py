# Copyright © 2026 Apple Inc.

import unittest
from types import SimpleNamespace

import mlx.core as mx
import test_models
from mlx.utils import tree_flatten, tree_map
from model_parallel_tests import Group

from mlx_lm.chat_templates.muse_spark import RESPONSE_FORMAT
from mlx_lm.generate import TextStateMachine
from mlx_lm.models import muse_spark
from mlx_lm.models.cache import RotatingKVCache

# Layers 0-2 are sliding (window 8) with RoPE, layer 3 is full and NoPE.
CONFIG = {
    "model_type": "muse_spark",
    # The default id is not in this vocab.
    "image_token_id": 255,
    "text_config": {
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "layer_rope_theta": [10000.0, 10000.0, 10000.0, 0],
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "sliding_window": 8,
        "final_logit_softcapping": 20.0,
        "output_multiplier": 2**-0.5,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_hidden_size": 32,
        "moe_intermediate_size": 32,
        "routed_scaling_factor": 1.5,
    },
}

# Tiles of 4 x 4 patches; layer 0 attends in tiles, layer 1 to all patches.
VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "patch_size": 2,
    "merge_size": 2,
    "pos_emb_height": 4,
    "pos_emb_width": 4,
    "layer_types": ["window_attention", "full_attention"],
}


def create_model(vision: bool = False) -> muse_spark.Model:
    mx.random.seed(0)
    config = CONFIG
    if vision:
        config = {**CONFIG, "vision_config": VISION_CONFIG, "projector_hidden_size": 48}
    args = muse_spark.ModelArgs.from_dict(config)
    return muse_spark.Model(args)


def image_prompt(
    model: muse_spark.Model, *, h: int, w: int
) -> tuple[mx.array, mx.array]:
    """An h x w image and a prompt with a patch token for each of its rows."""
    vision = model.args.vision
    side = vision.patch_size * vision.merge_size
    rows = (h // side) * (w // side)
    image = mx.random.uniform(shape=(3, h, w))
    inputs = mx.array([1, 2] + [model.args.image_token_id] * rows + [3, 4])
    return image, inputs


def rotate_half_rows(w: mx.array, head_dim: int) -> mx.array:
    """The q and k rows of each head in the Hugging Face order."""
    shape = w.shape
    w = w.reshape(-1, head_dim // 2, 2, *shape[1:])
    return mx.swapaxes(w, 1, 2).reshape(shape)


def hf_checkpoint(model: muse_spark.Model, weights: dict) -> dict:
    """The weights of model as a Hugging Face checkpoint stores them."""
    args = model.args.text
    q_dim = args.num_attention_heads * args.head_dim
    kv_dim = args.num_key_value_heads * args.head_dim
    hf = {}
    for k, v in weights.items():
        if k.endswith("self_attn.qkvg_proj.weight"):
            attn = k[: -len("qkvg_proj.weight")]
            parts = mx.split(v, [q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim])
            for name, part in zip(("q", "k", "v", "gate"), parts):
                if name in "qk":
                    part = rotate_half_rows(part, args.head_dim)
                hf[f"{attn}{name}_proj.weight"] = part
        elif ".attn.qkv_proj." in k:
            for name, part in zip("qkv", mx.split(v, 3)):
                if name in "qk":
                    part = rotate_half_rows(part, model.args.vision.head_dim)
                hf[k.replace("qkv_proj", f"{name}_proj")] = part
        elif k.endswith(("_alpha", "_beta")):
            # Zero gates give the default coefficients.
            name = "post_attention" if ".add_attn_" in k else "post_feedforward"
            prefix = k.rsplit(".", 1)[0]
            hf[f"{prefix}.{name}_residual_gate.gate"] = mx.zeros(v.shape)
        elif k.endswith(muse_spark._ZERO_CENTERED_NORMS):
            hf[k] = v - 1
        elif k.endswith("mlp.experts.gate_proj.weight"):
            up = weights[k.replace("gate_proj", "up_proj")]
            hf[k.replace("gate_proj.weight", "gate_up_proj")] = mx.concatenate(
                [v, up], axis=1
            )
        elif k.endswith("mlp.experts.up_proj.weight"):
            continue
        elif k.endswith("mlp.experts.down_proj.weight"):
            hf[k.replace("down_proj.weight", "down_proj")] = v
        else:
            hf[k] = v
    return hf


class TestMuseSpark(unittest.TestCase):
    def test_forward(self):
        model = create_model()
        for t in [mx.float32, mx.bfloat16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))
            # The soft-capped logits are fp32 for all weight dtypes.
            outputs = model(mx.array([[0, 1]]))
            self.assertEqual(outputs.shape, (1, 2, model.args.text.vocab_size))
            self.assertEqual(outputs.dtype, mx.float32)

    def test_vision_layer_types(self):
        # As transformers: full attention in every 4th and the last layer.
        vision = {**VISION_CONFIG, "num_hidden_layers": 6, "layer_types": None}
        window, full = "window_attention", "full_attention"
        self.assertEqual(
            muse_spark.VisionArgs.from_dict(vision).layer_types,
            [window, window, window, full, window, full],
        )

    def test_left_padding(self):
        # Its prompts fit in the sliding window.
        test_models.TestModels.left_padding_test_runner(self, create_model())

    def test_cached_matches_forward(self):
        model = create_model()
        tokens = mx.random.randint(0, model.args.text.vocab_size, (2, 16))
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
        norm = "model.language_model.layers.0.pre_feedforward_layernorm.weight"
        raw = {norm: mx.zeros((64,)), "lm_head.weight": weights["lm_head.weight"]}
        self.assertTrue(mx.array_equal(model.sanitize(raw)[norm], mx.ones((64,))))

    def test_sanitize_hf_weights(self):
        model = create_model(vision=True)
        weights = dict(tree_flatten(model.parameters()))
        hf = hf_checkpoint(model, weights)
        hf["model.audio_tower.norm.weight"] = mx.zeros((1,))
        sanitized = model.sanitize(hf)
        self.assertEqual(sanitized.keys(), weights.keys())
        for k, v in weights.items():
            self.assertTrue(mx.allclose(sanitized[k], v), k)

    def test_pipeline_layer_indices(self):
        # Rank 1 has layers 0-1 (sliding), rank 0 has 2 (sliding) and 3 (full).
        for rank, fa_idx, sliding in ((0, 1, [True, False]), (1, 0, [True, True])):
            model = create_model()
            model.model.pipeline(Group(rank, 2))
            text = model.model.language_model
            self.assertEqual(text.fa_idx, fa_idx)
            self.assertEqual(text.swa_idx, 0)
            caches = model.make_cache()
            self.assertEqual([isinstance(c, RotatingKVCache) for c in caches], sliding)

    def test_embed_inputs(self):
        model = create_model(vision=True)
        # 6 x 8 patches, more than one tile, give 3 x 4 rows.
        image, inputs = image_prompt(model, h=12, w=16)
        rows = model.encode_image(image)
        self.assertEqual(rows.shape, (12, model.args.text.hidden_size))
        h = model.embed_inputs(inputs, pixel_values=[image])
        text = model.model.language_model.embed(inputs)
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


class TestMuseSparkResponseFormat(unittest.TestCase):
    def split(self, text, step):
        tokenizer = SimpleNamespace(
            has_thinking=False,
            has_tool_calling=True,
            tool_call_start="<atem:function_calls>",
            tool_call_end="</atem:function_calls>",
            structural_markers=(),
            response_transitions=RESPONSE_FORMAT["transitions"],
        )
        reader = TextStateMachine.from_tokenizer(tokenizer).reader(
            state=RESPONSE_FORMAT["initial_state"]
        )
        delta = TextStateMachine.Delta()
        for i in range(0, len(text), step):
            delta += reader.feed(text[i : i + step])
        return delta + reader.finish()

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
                self.assertEqual(out.reasoning, "think")
                self.assertEqual(out.content, "onetwo")
                self.assertEqual(
                    out.tool_calls,
                    (
                        call[
                            len("<atem:function_calls>") : -len(
                                "</atem:function_calls>"
                            )
                        ],
                    ),
                )

    def test_quoted_call(self):
        # A call starts its own message; one in the answer is only text.
        call = (
            '<atem:function_calls><atem:invoke name="grep">'
            "</atem:invoke></atem:function_calls>"
        )
        text = f" to=user<|message|>Write {call} to search."
        for step in (1, 3, 7, len(text)):
            out = self.split(text, step)
            self.assertEqual(out.content, f"Write {call} to search.")
            self.assertEqual(out.tool_calls, ())


if __name__ == "__main__":
    unittest.main()
