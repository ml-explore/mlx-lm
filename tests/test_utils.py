# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm import convert, utils

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir_fid.name)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_load(self):
        from mlx_lm.models.qwen2 import Model as Qwen2Model

        model, _ = utils.load(HF_MODEL_PATH)
        self.assertIsInstance(model, Qwen2Model)

        model_lazy, _ = utils.load(HF_MODEL_PATH, lazy=True)

        mx.eval(model_lazy.parameters())

        p1 = model.layers[0].mlp.up_proj.weight
        p2 = model_lazy.layers[0].mlp.up_proj.weight
        self.assertTrue(mx.allclose(p1, p2))

    def test_make_shards(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=32,
            intermediate_size=4096,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=30_000,
        )
        model = llama.Model(args)
        weights = tree_flatten(model.parameters())
        gb = sum(p.nbytes for _, p in weights) // 2**30
        shards = utils.make_shards(dict(weights), 1)
        self.assertTrue(gb <= len(shards) <= gb + 1)

    def test_quantize(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = llama.Model(args)
        model, config = utils.quantize_model(model, {}, 64, 4)
        weights = dict(tree_flatten(model.parameters()))
        self.assertTrue("model.layers.2.mlp.up_proj.scales" in weights)
        self.assertTrue("model.layers.2.mlp.up_proj.biases" in weights)
        self.assertEqual(config["quantization"]["group_size"], 64)
        self.assertEqual(config["quantization"]["bits"], 4)

    def test_convert(self):
        mlx_path = os.path.join(self.test_dir, "mlx_model")

        convert(HF_MODEL_PATH, mlx_path=mlx_path, quantize=False)
        model, _ = utils.load(mlx_path)
        self.assertTrue(isinstance(model.layers[0].mlp.up_proj, nn.QuantizedLinear))
        self.assertTrue(isinstance(model.layers[-1].mlp.up_proj, nn.QuantizedLinear))

        # Check model weights have right type
        mlx_path = os.path.join(self.test_dir, "mlx_model_bf16")
        convert(HF_MODEL_PATH, mlx_path=mlx_path, dtype="bfloat16")
        model, _ = utils.load(mlx_path)

        self.assertEqual(model.layers[0].mlp.up_proj.scales.dtype, mx.bfloat16)
        self.assertEqual(model.layers[-1].mlp.up_proj.scales.dtype, mx.bfloat16)

    def test_load_model_with_custom_get_classes(self):
        class CustomQwenModel(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.config = args
                self.custom_attribute = "This is a custom model"

            def load_weights(self, weights, **kwargs):
                self.qwenWeights = weights

        class CustomQwenConfig:
            @classmethod
            def from_dict(cls, config):
                instance = cls()
                for k, v in config.items():
                    setattr(instance, k, v)
                return instance

        def custom_get_classes(config):
            return CustomQwenModel, CustomQwenConfig

        model_path = utils.hf_repo_to_path(HF_MODEL_PATH)
        model, _ = utils.load_model(model_path, get_model_classes=custom_get_classes)

        self.assertIsInstance(model, CustomQwenModel)
        self.assertTrue(hasattr(model, "custom_attribute"))
        self.assertEqual(model.custom_attribute, "This is a custom model")
        self.assertTrue(hasattr(model, "qwenWeights"))

    def test_load_model_gemma4_with_per_layer_projection_quantization(self):
        from mlx_lm.models import gemma4

        args = gemma4.ModelArgs.from_dict(
            {
                "model_type": "gemma4",
                "vocab_size": 32,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "layer_types": ["full_attention", "full_attention"],
                    "hidden_size_per_layer_input": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "tie_word_embeddings": True,
                },
            }
        )
        model = gemma4.Model(args)
        model, config = utils.quantize_model(
            model,
            {
                "model_type": "gemma4",
                "vocab_size": args.vocab_size,
                "text_config": args.text_config,
            },
            group_size=32,
            bits=4,
        )

        config["quantization"]["language_model.model.per_layer_model_projection"] = {
            "group_size": 32,
            "bits": 4,
        }

        with tempfile.TemporaryDirectory(dir=self.test_dir) as mlx_path:
            utils.save_model(mlx_path, model)
            utils.save_config(config, os.path.join(mlx_path, "config.json"))

            loaded, loaded_config = utils.load_model(Path(mlx_path))

            self.assertIn(
                "language_model.model.per_layer_model_projection",
                loaded_config["quantization"],
            )

            logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32))
            mx.eval(logits)
            self.assertEqual(logits.shape, (1, 3, args.vocab_size))

    def _awq_lm_head_triple(self, prefix=""):
        return {
            f"{prefix}lm_head.qweight": mx.zeros((4, 1), dtype=mx.int32),
            f"{prefix}lm_head.qzeros": mx.zeros((1, 1), dtype=mx.int32),
            f"{prefix}lm_head.scales": mx.zeros((1, 8), dtype=mx.float16),
        }

    def _qwen2_args(self, **overrides):
        from mlx_lm.models import qwen2

        base = dict(
            model_type="qwen2",
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-6,
            vocab_size=8,
        )
        base.update(overrides)
        return qwen2.ModelArgs.from_dict(base)

    def _gemma3_text_model(self):
        from mlx_lm.models import gemma3_text

        return gemma3_text.Model(
            gemma3_text.ModelArgs(
                model_type="gemma3_text",
                hidden_size=8,
                num_hidden_layers=1,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                rms_norm_eps=1e-6,
                vocab_size=8,
            )
        )

    def _recurrent_gemma_model(self):
        from mlx_lm.models import recurrent_gemma

        return recurrent_gemma.Model(
            recurrent_gemma.ModelArgs(
                model_type="recurrent_gemma",
                attention_bias=False,
                conv1d_width=4,
                hidden_size=8,
                intermediate_size=16,
                logits_soft_cap=30.0,
                num_attention_heads=2,
                num_hidden_layers=1,
                num_key_value_heads=1,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                attention_window_size=8,
                vocab_size=8,
                block_types=["attention"],
            )
        )

    def _qwen2_vl_model(self, *, tie_word_embeddings):
        from mlx_lm.models import qwen2_vl

        return qwen2_vl.Model(
            qwen2_vl.ModelArgs(
                model_type="qwen2_vl",
                text_config=dict(
                    model_type="qwen2",
                    hidden_size=8,
                    num_hidden_layers=1,
                    intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-6,
                    vocab_size=8,
                    tie_word_embeddings=tie_word_embeddings,
                ),
            )
        )

    def test_awq_drops_tied_lm_head_triple(self):
        from mlx_lm.models import qwen2

        model = qwen2.Model(self._qwen2_args(tie_word_embeddings=True))
        weights = {
            **self._awq_lm_head_triple(),
            "model.embed_tokens.weight": mx.zeros((8, 8), dtype=mx.float16),
        }
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in self._awq_lm_head_triple():
            self.assertNotIn(k, out)
        self.assertIn("model.embed_tokens.weight", out)

    def test_awq_preserves_untied_lm_head_triple(self):
        from mlx_lm.models import qwen2

        model = qwen2.Model(self._qwen2_args(tie_word_embeddings=False))
        weights = self._awq_lm_head_triple()
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in weights:
            self.assertIn(k, out)

    def test_awq_drops_when_tied_field_defaulted_in_modelargs(self):
        # Qwen2/Llama ModelArgs default `tie_word_embeddings=True`. A
        # checkpoint config that omits the field still produces a tied
        # model after `from_dict`. Because Qwen2 ModelArgs declares the
        # field, the missing parameter target is authoritative even
        # when the raw config is silent.
        from mlx_lm.models import qwen2

        args = qwen2.ModelArgs.from_dict(
            dict(
                model_type="qwen2",
                hidden_size=8,
                num_hidden_layers=1,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                rms_norm_eps=1e-6,
                vocab_size=8,
            )
        )
        self.assertTrue(args.tie_word_embeddings)
        model = qwen2.Model(args)
        out = utils._maybe_drop_redundant_lm_head_awq_triple(
            self._awq_lm_head_triple(), model, {}
        )
        for k in self._awq_lm_head_triple():
            self.assertNotIn(k, out)

    def test_awq_drops_prefixed_lm_head_triple_for_tied_vl(self):
        # Qwen2-VL sanitize prefixes text weights with `language_model.`
        # before AWQ processing runs. The triple lands at
        # `language_model.lm_head.{qweight,qzeros,scales}` which an
        # exact-key filter would miss.
        model = self._qwen2_vl_model(tie_word_embeddings=True)
        weights = self._awq_lm_head_triple(prefix="language_model.")
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in weights:
            self.assertNotIn(k, out)

    def test_awq_preserves_prefixed_lm_head_triple_for_untied_vl(self):
        model = self._qwen2_vl_model(tie_word_embeddings=False)
        weights = self._awq_lm_head_triple(prefix="language_model.")
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in weights:
            self.assertIn(k, out)

    def test_awq_drops_prefixed_when_text_config_overrides_top_level(self):
        # Multimodal config can have wrapper-level
        # `tie_word_embeddings: false` while `text_config` declares the
        # text submodel tied. Qwen2-VL builds its language model from
        # `args.text_config`, so the language submodel is tied per
        # text_config; the wrapper-level flag does not govern it.
        # `language_model.lm_head.qweight` must be dropped.
        model = self._qwen2_vl_model(tie_word_embeddings=True)
        weights = self._awq_lm_head_triple(prefix="language_model.")
        config = {
            "tie_word_embeddings": False,
            "text_config": {"tie_word_embeddings": True},
        }
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, config)
        for k in weights:
            self.assertNotIn(k, out)

    def test_awq_drops_for_gemma3_with_explicit_tied_config(self):
        # gemma3_text and recurrent_gemma decide tying by inspecting
        # weight content rather than a config field; sanitize pops
        # `lm_head` when `lm_head.weight` is absent. For a tied
        # checkpoint with a redundant AWQ triple, an explicit
        # `tie_word_embeddings: true` in config gates the drop.
        model = self._gemma3_text_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        out = utils._maybe_drop_redundant_lm_head_awq_triple(
            weights, model, {"tie_word_embeddings": True}
        )
        for k in self._awq_lm_head_triple():
            self.assertNotIn(k, out)
        self.assertTrue(model.tie_word_embeddings)
        self.assertNotIn("lm_head", model)

    def test_awq_preserves_for_gemma3_with_silent_config(self):
        # Without an explicit tied signal AND with `ModelArgs` lacking
        # the `tie_word_embeddings` field, the missing parameter target
        # is ambiguous between a tied checkpoint with a redundant
        # triple and an untied checkpoint with a real quantized output
        # head. Silently dropping the latter would produce wrong logits;
        # preserving lets strict load fail loudly.
        model = self._gemma3_text_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in self._awq_lm_head_triple():
            self.assertIn(k, out)

    def test_awq_preserves_for_gemma3_with_explicit_untied_config(self):
        # Authoritative untied: never drop, even when the model has no
        # parameter target (sanitize already mishandled it). Strict
        # load will fail loudly afterward, surfacing the gemma3_text
        # sanitize limitation rather than silently using tied
        # embeddings.
        model = self._gemma3_text_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        out = utils._maybe_drop_redundant_lm_head_awq_triple(
            weights, model, {"tie_word_embeddings": False}
        )
        for k in self._awq_lm_head_triple():
            self.assertIn(k, out)

    def test_awq_drops_for_recurrent_gemma_with_explicit_tied_config(self):
        model = self._recurrent_gemma_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        out = utils._maybe_drop_redundant_lm_head_awq_triple(
            weights, model, {"tie_word_embeddings": True}
        )
        for k in self._awq_lm_head_triple():
            self.assertNotIn(k, out)
        self.assertNotIn("lm_head", model)

    def test_awq_preserves_for_recurrent_gemma_with_silent_config(self):
        model = self._recurrent_gemma_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in self._awq_lm_head_triple():
            self.assertIn(k, out)

    def _gpt2_model(self):
        from mlx_lm.models import gpt2

        return gpt2.Model(
            gpt2.ModelArgs(
                model_type="gpt2",
                n_ctx=8,
                n_embd=8,
                n_head=2,
                n_layer=1,
                n_positions=8,
                layer_norm_epsilon=1e-5,
                vocab_size=8,
            )
        )

    def test_awq_drops_model_prefixed_for_gpt2_with_explicit_tied_config(self):
        # gpt2 (and gpt_neox) sanitize prefixes every key with `model.`
        # so a top-level `lm_head.qweight` becomes
        # `model.lm_head.qweight`. The constructed wrapper has no
        # `model.lm_head` target (the head is tied to `model.wte`),
        # and the governing tying signal lives at top-level config —
        # the `model.` prefix is just a sanitize-level renaming, not
        # a multimodal submodel boundary. With explicit
        # `tie_word_embeddings: true`, drop the prefixed triple.
        model = self._gpt2_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        # Sanity: gpt2 sanitize re-namespaced under `model.`.
        self.assertIn("model.lm_head.qweight", weights)
        out = utils._maybe_drop_redundant_lm_head_awq_triple(
            weights, model, {"tie_word_embeddings": True}
        )
        for k in self._awq_lm_head_triple(prefix="model."):
            self.assertNotIn(k, out)

    def test_awq_preserves_model_prefixed_for_gpt2_with_silent_config(self):
        # gpt2 ModelArgs has no `tie_word_embeddings` field, so a
        # silent config leaves the redundant triple ambiguous; preserve
        # and let strict load fail loudly rather than silently produce
        # wrong logits.
        model = self._gpt2_model()
        weights = model.sanitize(self._awq_lm_head_triple())
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in self._awq_lm_head_triple(prefix="model."):
            self.assertIn(k, out)

    def _gemma4_config(self, *, tie_word_embeddings):
        return {
            "model_type": "gemma4",
            "vocab_size": 8,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 4,
                "sliding_window": 8,
                "sliding_window_pattern": 1,
                "layer_types": ["full_attention"],
                "hidden_size_per_layer_input": 8,
                "vocab_size_per_layer_input": 8,
                "num_kv_shared_layers": 0,
                "tie_word_embeddings": tie_word_embeddings,
            },
        }

    def test_awq_drops_language_model_model_prefixed_for_tied_gemma4(self):
        # gemma4.sanitize re-namespaces `model.language_model.X` to
        # `language_model.model.X` (a sanitize-level renaming inside
        # the text wrapper, not a separate tying boundary). The
        # tying decision still belongs to the text submodel, governed
        # by `config["text_config"]["tie_word_embeddings"]`.
        from mlx_lm.models import gemma4

        config = self._gemma4_config(tie_word_embeddings=True)
        model = gemma4.Model(gemma4.ModelArgs.from_dict(config))
        weights = model.sanitize(
            self._awq_lm_head_triple(prefix="model.language_model.")
        )
        # Sanity: gemma4 sanitize moved keys under `language_model.model.`.
        self.assertIn("language_model.model.lm_head.qweight", weights)
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, config)
        for k in self._awq_lm_head_triple(prefix="language_model.model."):
            self.assertNotIn(k, out)

    def test_awq_preserves_language_model_model_prefixed_for_untied_gemma4(self):
        # Mirror: with text_config explicitly untied, `language_model.
        # model.lm_head.{qweight,qzeros,scales}` must NOT be dropped —
        # the sanitize prefix is irrelevant to the tying decision, and
        # silently dropping a real quantized output head would produce
        # wrong logits.
        from mlx_lm.models import gemma4

        config = self._gemma4_config(tie_word_embeddings=False)
        model = gemma4.Model(gemma4.ModelArgs.from_dict(config))
        weights = model.sanitize(
            self._awq_lm_head_triple(prefix="model.language_model.")
        )
        self.assertIn("language_model.model.lm_head.qweight", weights)
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, config)
        for k in self._awq_lm_head_triple(prefix="language_model.model."):
            self.assertIn(k, out)

    def test_awq_preserves_for_qwen3_5_with_silent_default_untied(self):
        # qwen3_5 TextModelArgs defaults `tie_word_embeddings=False`,
        # i.e. defaulted-untied. The AWQ ckpt with a real quantized
        # output head ships at `model.language_model.lm_head.*`
        # (HF flat layout); qwen3_5 sanitize re-namespaces to
        # `language_model.model.lm_head.*`. With silent config and a
        # field that exists but holds False, the helper must use the
        # actual value (not just the field's existence) to decide:
        # preserve the triple and let strict load fail loudly rather
        # than silently dropping a real output head.
        from mlx_lm.models import qwen3_5

        config = {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "head_dim": 4,
                "full_attention_interval": 4,
                "rope_parameters": {
                    "type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.25,
                },
            },
        }
        model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
        # Sanity: TextModel default-untied, lm_head module exists at
        # `language_model.lm_head` (NOT `language_model.model.lm_head`).
        self.assertFalse(model.language_model.args.tie_word_embeddings)
        weights = model.sanitize(
            self._awq_lm_head_triple(prefix="model.language_model.")
        )
        self.assertIn("language_model.model.lm_head.qweight", weights)
        out = utils._maybe_drop_redundant_lm_head_awq_triple(weights, model, {})
        for k in self._awq_lm_head_triple(prefix="language_model.model."):
            self.assertIn(k, out)


if __name__ == "__main__":
    unittest.main()
