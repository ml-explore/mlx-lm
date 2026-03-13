import importlib
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, mtp_generate_step
from mlx_lm.models.cache import make_prompt_cache


def _make_qwen3_5_mtp_model():
    """Create a tiny Qwen3.5 model with an MTP head for testing."""
    module = importlib.import_module("mlx_lm.models.qwen3_5")
    args = module.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
            },
        }
    )
    model = module.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


class TestMTP(unittest.TestCase):
    """Tests for native MTP (Multi-Token Prediction) speculative decoding.

    Uses a tiny synthetic Qwen3.5 model (4 layers, hidden=64, vocab=256)
    with mtp_num_hidden_layers=1 and full_attention_interval=2, giving a
    mix of GatedDeltaNet (SSM) and full-attention layers.

    Not tested here (would require a real tokenizer loaded from HF):
    - stream_generate() with mtp=True/False flag dispatch
    - Server integration (--mtp flag, is_batchable)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _make_qwen3_5_mtp_model()

    def test_mtp_module_exists(self):
        """Model with mtp_num_hidden_layers=1 should have MTP head."""
        self.assertTrue(hasattr(self.model, "mtp_forward"))
        self.assertTrue(hasattr(self.model, "make_mtp_cache"))
        lm = self.model.language_model
        self.assertTrue(hasattr(lm, "mtp"))
        self.assertEqual(len(lm.mtp.layers), 1)

    def test_make_mtp_cache(self):
        """make_mtp_cache should return one KVCache per MTP layer."""
        mtp_cache = self.model.make_mtp_cache()
        self.assertEqual(len(mtp_cache), 1)
        self.assertTrue(mtp_cache[0].is_trimmable())

    def test_return_hidden(self):
        """return_hidden=True should return (logits, hidden) with correct shapes."""
        inputs = mx.array([[0, 1, 2]])
        cache = make_prompt_cache(self.model)
        out, hidden = self.model(inputs, cache=cache, return_hidden=True)
        self.assertEqual(out.shape, (1, 3, 256))
        self.assertEqual(hidden.shape, (1, 3, 64))

    def test_mtp_forward_shape(self):
        """mtp_forward should produce logits of shape (B, 1, vocab)."""
        hidden = mx.random.normal((1, 1, 64))
        next_ids = mx.array([[5]])
        mtp_cache = self.model.make_mtp_cache()
        logits = self.model.mtp_forward(hidden, next_ids, mtp_cache)
        self.assertEqual(logits.shape, (1, 1, 256))

    def test_hidden_is_pre_norm(self):
        """Hidden states returned with return_hidden should be pre-norm.

        This verifies the fix for double normalization: the backbone returns
        pre-norm hidden states, and the final norm is applied only before
        lm_head (not before the MTP head).
        """
        lm = self.model.language_model
        inputs = mx.array([[0, 1, 2]])
        cache = make_prompt_cache(self.model)

        _, hidden = lm(inputs, cache=cache, return_hidden=True)

        # Apply the final norm manually and check it changes the values.
        normed = lm.model.norm(hidden)
        self.assertFalse(mx.allclose(hidden, normed, atol=1e-5).item())

    def test_quant_predicate_excludes_mtp_fc(self):
        """quant_predicate should exclude mtp.fc from quantization."""
        lm = self.model.language_model
        predicate = lm.quant_predicate
        self.assertIsNotNone(predicate)
        # mtp.fc should not be quantized
        self.assertFalse(predicate("mtp.fc", None))
        # Regular layers should be quantized
        self.assertTrue(predicate("layers.0.mlp.gate_proj", None))

    def test_mtp_generate_identity(self):
        """mtp_generate_step should produce the same greedy tokens as generate_step.

        This is the most important correctness test: it proves that the
        draft/verify loop, SSM state rollback on rejection, and MTP cache
        management are all correct.  Any bug in these would cause the MTP
        path to diverge from standard generation.
        """
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 10

        def greedy(logprobs):
            return mx.argmax(logprobs, axis=-1)

        # Standard generation
        std_cache = make_prompt_cache(self.model)
        std_tokens = []
        for i, (tok, _) in enumerate(
            generate_step(prompt, self.model, sampler=greedy, prompt_cache=std_cache)
        ):
            std_tokens.append(int(tok))
            if i + 1 >= n_tokens:
                break

        # MTP generation
        mtp_tokens = []
        for tok, _, _ in mtp_generate_step(
            prompt, self.model, sampler=greedy, max_tokens=n_tokens
        ):
            mtp_tokens.append(int(tok))
            if len(mtp_tokens) >= n_tokens:
                break

        self.assertEqual(
            std_tokens,
            mtp_tokens,
            f"Token mismatch: std={std_tokens}, mtp={mtp_tokens}",
        )

    def test_mtp_generate_runs(self):
        """mtp_generate_step should complete without errors.

        Exercises the full end-to-end path: prefill, backbone forward with
        return_hidden, MTP draft generation, verification with n_confirmed,
        SSM rollback on rejection, and MTP cache trimming.
        """
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 10

        def greedy(logprobs):
            return mx.argmax(logprobs, axis=-1)

        tokens = []
        for tok, _, _ in mtp_generate_step(
            prompt, self.model, sampler=greedy, max_tokens=n_tokens
        ):
            tokens.append(int(tok))
            if len(tokens) >= n_tokens:
                break

        self.assertEqual(len(tokens), n_tokens)


if __name__ == "__main__":
    unittest.main()
