# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import prism_hadamard_qwen35 as phq
from mlx_lm.models import qwen3_5
from mlx_lm.utils import load_model

BLOCK = 512

TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": 256,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "rms_norm_eps": 1e-6,
    "head_dim": 64,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.25,
    "max_position_embeddings": 1000,
}

TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def exact_packed(rows, width, key):
    """Random 2-bit affine weights that dequantize exactly."""
    k1, k2 = mx.random.split(key)
    codes = mx.random.randint(0, 4, (rows, width), key=k1).astype(mx.uint32)
    words = codes.reshape(rows, width // 16, 16) << (2 * mx.arange(16, dtype=mx.uint32))
    weight = words.sum(axis=-1).astype(mx.uint32)
    scales = (mx.random.uniform(0.01, 0.05, (rows, width // 128), key=k2)).astype(
        mx.float16
    )
    biases = (-1.5 * scales).astype(mx.float16)
    return weight, scales, biases


def dequant(weight, scales, biases):
    return mx.dequantize(weight, scales, biases, group_size=128, bits=2).astype(
        mx.float32
    )


def random_signs(width, key):
    return mx.where(mx.random.bernoulli(0.5, (width,), key=key), 1.0, -1.0)


class TestHadamardLayers(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def test_linear_matches_unfolded_weights(self):
        rows, width = 192, 1024
        weight, scales, biases = exact_packed(rows, width, mx.random.key(1))
        signs = random_signs(width, mx.random.key(2))
        layer = phq.HadamardQuantizedLinear(width, rows, BLOCK)
        layer.update(
            {"weight": weight, "scales": scales, "biases": biases, "signs": signs}
        )

        # Folded W_f = W D H; the unfolded weight is W = W_f H D.
        unfolded = phq.hadamard_rotate(
            dequant(weight, scales, biases), BLOCK, signs, inverse=True
        )
        x = mx.random.normal((3, 5, width)).astype(mx.float16)
        expected = x.astype(mx.float32) @ unfolded.T
        out = layer(x).astype(mx.float32)
        self.assertTrue(mx.allclose(out, expected, atol=2e-2, rtol=2e-2).item())

    def test_embedding_and_tied_head(self):
        vocab, width = 64, 512
        weight, scales, biases = exact_packed(vocab, width, mx.random.key(3))
        signs = random_signs(width, mx.random.key(4))
        layer = phq.HadamardQuantizedEmbedding(width, vocab, BLOCK)
        layer.update(
            {"weight": weight, "scales": scales, "biases": biases, "signs": signs}
        )
        table = phq.hadamard_rotate(
            dequant(weight, scales, biases), BLOCK, signs, inverse=True
        )

        ids = mx.array([[0, 5, 63], [7, 7, 1]])
        self.assertTrue(
            mx.allclose(layer(ids).astype(mx.float32), table[ids], atol=1e-2).item()
        )

        h = mx.random.normal((2, width)).astype(mx.float16)
        logits = layer.as_linear(h).astype(mx.float32)
        self.assertTrue(
            mx.allclose(
                logits, h.astype(mx.float32) @ table.T, atol=2e-2, rtol=2e-2
            ).item()
        )

    def test_rejects_bad_block(self):
        with self.assertRaises(ValueError):
            phq.HadamardQuantizedLinear(1024, 64, 384)
        with self.assertRaises(ValueError):
            phq.HadamardQuantizedLinear(768, 64, 512)


def build_pack(schema, tied):
    """A packed checkpoint plus a stock qwen3_5 model holding the unfolded weights."""
    text = dict(TEXT_CONFIG, tie_word_embeddings=tied)
    ref = qwen3_5.Model(
        qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text})
    )
    lm = ref.language_model
    paths = [
        f"model.layers.{i}.{t}"
        for i, layer in enumerate(lm.model.layers)
        for t in TARGETS
        if _has(layer, t)
    ]
    paths.append("model.embed_tokens")
    if not tied:
        paths.append("lm_head")

    named = dict(lm.named_modules())
    packed, modules, unfolded = {}, [], {}
    for n, path in enumerate(paths):
        mod = named[path]
        rows, width = mod.weight.shape
        weight, scales, biases = exact_packed(rows, width, mx.random.key(100 + n))
        signs = random_signs(width, mx.random.key(500 + (width % 997)))
        unfolded[path] = phq.hadamard_rotate(
            dequant(weight, scales, biases), BLOCK, signs, inverse=True
        )
        packed.update(
            {
                f"{path}.weight": weight,
                f"{path}.scales": scales,
                f"{path}.biases": biases,
                f"{path}.signs": signs,
            }
        )
        modules.append(
            {
                "path": path,
                "block": BLOCK,
                "embedding": path == "model.embed_tokens",
                "dtype": "float16",
            }
        )

    lm.update_modules({})
    for path, w in unfolded.items():
        named[path].weight = w
    dense = {
        k: v
        for k, v in tree_flatten(lm.parameters())
        if not any(k == f"{p}.weight" for p in paths)
    }
    weights = {**dense, **packed}
    prefix = "" if schema == 1 else "language_model."
    weights = {prefix + k: v for k, v in weights.items()}
    if schema == 2:
        weights["vision_tower.patch_embed.proj.weight"] = mx.zeros((4, 4))

    config = {
        "model_type": "prism_hadamard_qwen35",
        "schema_version": schema,
        "tensor_namespace": phq._NAMESPACES[schema],
        "gdn_activation_layout": "grouped",
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "modules": modules,
        "text_config": text,
    }
    return config, weights, ref


def _has(layer, target):
    obj = layer
    for part in target.split("."):
        if part not in obj:
            return False
        obj = obj[part]
    return True


class TestHadamardModel(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def _check(self, schema, tied):
        config, weights, ref = build_pack(schema, tied)
        with tempfile.TemporaryDirectory() as d:
            mx.save_safetensors(str(Path(d) / "model.safetensors"), weights)
            (Path(d) / "config.json").write_text(json.dumps(config))
            model, _ = load_model(Path(d))
        self.assertIsInstance(model, phq.Model)
        packed = [
            m
            for _, m in model.named_modules()
            if isinstance(m, phq.HadamardQuantizedLinear)
        ]
        self.assertEqual(len(packed), len(config["modules"]))

        ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        out = model(ids).astype(mx.float32)
        expected = ref(ids).astype(mx.float32)
        self.assertEqual(out.shape, (1, 8, TEXT_CONFIG["vocab_size"]))
        err = mx.abs(out - expected).max().item() / (
            mx.abs(expected).max().item() + 1e-6
        )
        self.assertLess(err, 2e-2)

    def test_schema1_untied(self):
        self._check(1, tied=False)

    def test_schema1_tied(self):
        self._check(1, tied=True)

    def test_schema2_untied(self):
        self._check(2, tied=False)

    def test_rejects_wrong_namespace(self):
        config, _, _ = build_pack(1, tied=False)
        config["tensor_namespace"] = "mlx-vlm-qwen3_5"
        with self.assertRaises(ValueError):
            phq.ModelArgs.from_dict(config)

    def test_rejects_bad_signs(self):
        config, weights, _ = build_pack(1, tied=False)
        key = next(k for k in weights if k.endswith(".signs"))
        weights[key] = weights[key] * 2
        model = phq.Model(phq.ModelArgs.from_dict(config))
        with self.assertRaises(ValueError):
            model.sanitize(weights)


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    unittest.main()
