"""
Tests for LLaDA (Large Language Diffusion with mAsking) model implementation.
"""

import unittest

import mlx.core as mx
from mlx_lm.models import llada
from mlx_lm.llada_generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    generate,
)


class TestLLaDAModel(unittest.TestCase):
    """Tests for LLaDA model architecture."""

    @classmethod
    def setUpClass(cls):
        """Create a small LLaDA model for testing."""
        cls.args = llada.ModelArgs(
            model_type="llada",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            tie_word_embeddings=False,
            mask_token_id=999,
        )
        cls.model = llada.Model(cls.args)

    def test_model_instantiation(self):
        """Test that the model can be instantiated."""
        self.assertIsInstance(self.model, llada.Model)
        self.assertEqual(self.model.model_type, "llada")

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len = 2, 10
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * batch_size)

        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.args.vocab_size))

    def test_forward_pass_single_token(self):
        """Test forward pass with single token."""
        x = mx.array([[42]])
        output = self.model(x)
        self.assertEqual(output.shape, (1, 1, self.args.vocab_size))

    def test_forward_pass_with_cache(self):
        """Test forward pass with KV cache."""
        x = mx.array([[1, 2, 3, 4, 5]])
        cache = self.model.make_cache()

        output = self.model(x, cache=cache)

        self.assertEqual(output.shape, (1, 5, self.args.vocab_size))

    def test_layers_property(self):
        """Test that layers property returns transformer blocks."""
        layers = self.model.layers
        self.assertEqual(len(layers), self.args.num_hidden_layers)

    def test_make_cache(self):
        """Test cache creation."""
        cache = self.model.make_cache()
        self.assertEqual(len(cache), self.args.num_hidden_layers)


class TestLLaDAWeightSanitization(unittest.TestCase):
    """Tests for weight name remapping."""

    def setUp(self):
        """Create model for testing."""
        self.args = llada.ModelArgs(
            model_type="llada",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            tie_word_embeddings=False,
        )
        self.model = llada.Model(self.args)

    def test_sanitize_attention_weights(self):
        """Test attention weight name remapping."""
        weights = {
            "model.layers.0.attn_norm.weight": mx.zeros((64,)),
            "model.layers.0.attn_out.weight": mx.zeros((64, 64)),
            "model.layers.0.q_proj.weight": mx.zeros((64, 64)),
            "model.layers.0.k_proj.weight": mx.zeros((64, 64)),
            "model.layers.0.v_proj.weight": mx.zeros((64, 64)),
        }

        sanitized = self.model.sanitize(weights)

        self.assertIn("model.layers.0.input_layernorm.weight", sanitized)
        self.assertIn("model.layers.0.self_attn.o_proj.weight", sanitized)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertIn("model.layers.0.self_attn.k_proj.weight", sanitized)
        self.assertIn("model.layers.0.self_attn.v_proj.weight", sanitized)

    def test_sanitize_mlp_weights(self):
        """Test MLP weight name remapping."""
        weights = {
            "model.layers.0.ff_norm.weight": mx.zeros((64,)),
            "model.layers.0.ff_proj.weight": mx.zeros((128, 64)),
            "model.layers.0.ff_out.weight": mx.zeros((64, 128)),
            "model.layers.0.up_proj.weight": mx.zeros((128, 64)),
        }

        sanitized = self.model.sanitize(weights)

        self.assertIn("model.layers.0.post_attention_layernorm.weight", sanitized)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", sanitized)
        self.assertIn("model.layers.0.mlp.down_proj.weight", sanitized)
        self.assertIn("model.layers.0.mlp.up_proj.weight", sanitized)

    def test_sanitize_lm_head(self):
        """Test lm_head weight remapping."""
        weights = {
            "model.lm_head.weight": mx.zeros((1000, 64)),
        }

        sanitized = self.model.sanitize(weights)

        self.assertIn("lm_head.weight", sanitized)
        self.assertNotIn("model.lm_head.weight", sanitized)

    def test_sanitize_removes_rotary_emb(self):
        """Test that rotary embedding buffers are removed."""
        weights = {
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((32,)),
            "model.layers.0.q_proj.weight": mx.zeros((64, 64)),
        }

        sanitized = self.model.sanitize(weights)

        self.assertNotIn("model.layers.0.self_attn.rotary_emb.inv_freq", sanitized)


class TestLLaDAGenerate(unittest.TestCase):
    """Tests for LLaDA generation functions."""

    def test_add_gumbel_noise_zero_temperature(self):
        """Test that zero temperature returns logits unchanged."""
        logits = mx.array([[[1.0, 2.0, 3.0]]])

        result = add_gumbel_noise(logits, temperature=0.0)

        self.assertTrue(mx.allclose(result, logits).item())

    def test_add_gumbel_noise_nonzero_temperature(self):
        """Test that nonzero temperature adds noise."""
        mx.random.seed(42)
        logits = mx.array([[[1.0, 2.0, 3.0]]])

        result = add_gumbel_noise(logits, temperature=1.0)

        # Result should be different from input
        self.assertFalse(mx.allclose(result, logits).item())
        # Result should have same shape
        self.assertEqual(result.shape, logits.shape)

    def test_get_num_transfer_tokens(self):
        """Test token transfer calculation."""
        # 10 masked tokens, 5 steps -> 2 tokens per step
        mask_index = mx.array([[True] * 10])

        result = get_num_transfer_tokens(mask_index, steps=5)

        self.assertEqual(result.shape, (1, 5))
        self.assertEqual(result.sum().item(), 10)

    def test_get_num_transfer_tokens_with_remainder(self):
        """Test token transfer with non-divisible count."""
        # 7 masked tokens, 3 steps -> 3, 2, 2 tokens per step
        mask_index = mx.array([[True] * 7])

        result = get_num_transfer_tokens(mask_index, steps=3)

        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result.sum().item(), 7)

    def test_generate_basic(self):
        """Test basic generation."""
        args = llada.ModelArgs(
            model_type="llada",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            vocab_size=100,
            mask_token_id=99,
        )
        model = llada.Model(args)

        prompt = mx.array([[1, 2, 3]])

        output = generate(
            model,
            prompt,
            steps=4,
            gen_length=8,
            block_length=4,
            temperature=0.0,
            mask_id=99,
        )

        # Output should be prompt + generated tokens
        self.assertEqual(output.shape, (1, 11))
        # Prompt should be preserved
        self.assertTrue(mx.array_equal(output[0, :3], prompt[0]).item())

    def test_generate_output_shape(self):
        """Test that generation produces correct output shape."""
        args = llada.ModelArgs(
            model_type="llada",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            vocab_size=100,
            mask_token_id=99,
        )
        model = llada.Model(args)

        prompt = mx.array([[1, 2, 3]])
        mask_id = 99
        gen_length = 8

        output = generate(
            model,
            prompt,
            steps=8,
            gen_length=gen_length,
            block_length=4,
            temperature=0.0,
            mask_id=mask_id,
        )

        # Output shape should be (batch, prompt_len + gen_length)
        self.assertEqual(output.shape, (1, 3 + gen_length))

        # Prompt tokens should be preserved
        self.assertTrue(mx.array_equal(output[0, :3], prompt[0]).item())


class TestLLaDAModelArgs(unittest.TestCase):
    """Tests for LLaDA model arguments."""

    def test_default_kv_heads(self):
        """Test that num_key_value_heads defaults to num_attention_heads."""
        args = llada.ModelArgs(
            model_type="llada",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=8,
            vocab_size=1000,
        )

        self.assertEqual(args.num_key_value_heads, 8)

    def test_explicit_kv_heads(self):
        """Test explicit num_key_value_heads."""
        args = llada.ModelArgs(
            model_type="llada",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=1000,
        )

        self.assertEqual(args.num_key_value_heads, 2)


if __name__ == "__main__":
    unittest.main()
