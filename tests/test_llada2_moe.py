"""
Tests for LLaDA2 MoE (Mixture of Experts) model and generation.

Run with: PYTHONPATH=. python -m pytest tests/test_llada2_moe.py -v
"""

import unittest

import mlx.core as mx
from mlx_lm.models import llada2_moe
from mlx_lm import llada2_generate


class TestLLaDA2MoeModel(unittest.TestCase):
    """Tests for LLaDA2 MoE model architecture."""

    @classmethod
    def setUpClass(cls):
        """Create a small LLaDA2 MoE model for testing."""
        cls.args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            use_qk_norm=True,
            tie_word_embeddings=False,
            # MoE config
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            n_group=2,
            topk_group=1,
            moe_intermediate_size=32,
            first_k_dense_replace=1,  # First layer is dense
            routed_scaling_factor=1.0,
        )
        cls.model = llada2_moe.Model(cls.args)

    def test_model_instantiation(self):
        """Test that the model can be instantiated."""
        self.assertIsInstance(self.model, llada2_moe.Model)
        self.assertEqual(self.model.model_type, "llada2_moe")

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

    def test_layers_property(self):
        """Test that layers property returns decoder layers."""
        layers = self.model.layers
        self.assertEqual(len(layers), self.args.num_hidden_layers)

    def test_dense_vs_moe_layers(self):
        """Test that first layer is dense and rest are MoE."""
        layers = self.model.layers
        # First layer should be dense (first_k_dense_replace=1)
        self.assertFalse(layers[0].is_moe)
        # Remaining layers should be MoE
        for i in range(1, len(layers)):
            self.assertTrue(layers[i].is_moe)


class TestLLaDA2MoeAttention(unittest.TestCase):
    """Tests for LLaDA2 attention with partial rotary and QK norm."""

    def setUp(self):
        """Create attention module for testing."""
        self.args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            partial_rotary_factor=0.5,
            use_qk_norm=True,
        )
        self.attention = llada2_moe.Attention(self.args)

    def test_partial_rotary_dim(self):
        """Test that partial rotary dimension is correct."""
        expected_rotary_dim = int(self.args.head_dim * self.args.partial_rotary_factor)
        self.assertEqual(self.attention.rotary_dim, expected_rotary_dim)

    def test_qk_norm_layers_exist(self):
        """Test that QK normalization layers are created."""
        self.assertTrue(hasattr(self.attention, "query_layernorm"))
        self.assertTrue(hasattr(self.attention, "key_layernorm"))

    def test_attention_output_shape(self):
        """Test attention output shape."""
        x = mx.random.normal((2, 10, 64))
        output = self.attention(x)
        self.assertEqual(output.shape, (2, 10, 64))


class TestLLaDA2MoEGate(unittest.TestCase):
    """Tests for MoE gating mechanism."""

    def setUp(self):
        """Create MoE gate for testing."""
        self.args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=2.0,
        )
        self.gate = llada2_moe.MoEGate(self.args)

    def test_gate_output_shapes(self):
        """Test that gate returns correct shapes."""
        x = mx.random.normal((2, 10, 64))
        inds, scores = self.gate(x)

        # Should select top_k experts per token
        self.assertEqual(inds.shape, (2, 10, self.args.num_experts_per_tok))
        self.assertEqual(scores.shape, (2, 10, self.args.num_experts_per_tok))

    def test_gate_indices_valid(self):
        """Test that gate indices are valid expert indices."""
        x = mx.random.normal((2, 10, 64))
        inds, _ = self.gate(x)

        # All indices should be in range [0, num_experts)
        self.assertTrue(mx.all(inds >= 0).item())
        self.assertTrue(mx.all(inds < self.args.num_experts).item())


class TestLLaDA2MoEBlock(unittest.TestCase):
    """Tests for Sparse MoE block."""

    def setUp(self):
        """Create MoE block for testing."""
        self.args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=32,
            n_group=2,
            topk_group=1,
        )
        self.moe_block = llada2_moe.SparseMoEBlock(self.args)

    def test_moe_block_output_shape(self):
        """Test MoE block output shape."""
        x = mx.random.normal((2, 10, 64))
        output = self.moe_block(x)
        self.assertEqual(output.shape, x.shape)

    def test_shared_experts_exist(self):
        """Test that shared experts are created when specified."""
        self.assertIsNotNone(self.moe_block.shared_experts)


class TestLLaDA2Generate(unittest.TestCase):
    """Tests for LLaDA2 generation functions."""

    def test_get_num_transfer_tokens(self):
        """Test token transfer schedule calculation."""
        block_length = 10
        steps = 5

        schedule = llada2_generate.get_num_transfer_tokens(block_length, steps)

        # Should have one entry per step
        self.assertEqual(schedule.shape[0], steps)
        # Sum should equal block_length
        self.assertEqual(mx.sum(schedule).item(), block_length)

    def test_get_num_transfer_tokens_remainder(self):
        """Test token transfer schedule with remainder."""
        block_length = 7
        steps = 3

        schedule = llada2_generate.get_num_transfer_tokens(block_length, steps)

        # Sum should equal block_length
        self.assertEqual(mx.sum(schedule).item(), block_length)

    def test_create_block_diagonal_mask(self):
        """Test block-diagonal attention mask creation."""
        num_blocks = 3
        block_length = 4

        mask = llada2_generate.create_block_diagonal_mask(num_blocks, block_length)

        # Shape should be [1, 1, total_len, total_len]
        total_len = num_blocks * block_length
        self.assertEqual(mask.shape, (1, 1, total_len, total_len))

        # Check block-diagonal structure
        # Position (0,0) should be 0 (can attend)
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)
        # Position in block 1 attending to block 0 should be 0 (can attend)
        self.assertEqual(mask[0, 0, block_length, 0].item(), 0.0)
        # Position in block 0 attending to block 1 should be -inf (cannot attend)
        self.assertTrue(mask[0, 0, 0, block_length].item() == float("-inf"))

    def test_add_gumbel_noise_zero_temperature(self):
        """Test that zero temperature returns logits unchanged."""
        logits = mx.array([[[1.0, 2.0, 3.0]]])

        result = llada2_generate.add_gumbel_noise(logits, temperature=0.0)

        self.assertTrue(mx.allclose(result, logits).item())

    def test_add_gumbel_noise_nonzero_temperature(self):
        """Test that nonzero temperature adds noise."""
        mx.random.seed(42)
        logits = mx.array([[[1.0, 2.0, 3.0]]])

        result = llada2_generate.add_gumbel_noise(logits, temperature=1.0)

        # Result should be different from input
        self.assertFalse(mx.allclose(result, logits).item())
        # Result should have same shape
        self.assertEqual(result.shape, logits.shape)

    def test_sample_tokens_greedy(self):
        """Test greedy sampling (temperature=0)."""
        logits = mx.array([[[1.0, 2.0, 5.0, 3.0]]])  # Index 2 has highest

        tokens, probs = llada2_generate.sample_tokens(logits, temperature=0.0)

        self.assertEqual(tokens[0, 0].item(), 2)

    def test_generate_basic(self):
        """Test basic generation."""
        args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            first_k_dense_replace=0,
            n_group=2,
            topk_group=1,
        )
        model = llada2_moe.Model(args)
        mask_id = 999
        eos_id = 998

        prompt = mx.array([[1, 2, 3]])

        output = llada2_generate.generate(
            model,
            prompt,
            max_new_tokens=8,
            block_length=4,
            steps=4,
            temperature=0.0,
            mask_id=mask_id,
            eos_id=eos_id,
        )

        # Output should include prompt
        self.assertGreaterEqual(output.shape[1], prompt.shape[1])
        # Prompt should be preserved
        self.assertTrue(mx.array_equal(output[0, :3], prompt[0]).item())

    def test_generate_no_mask_tokens(self):
        """Test that output has no mask tokens."""
        args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            first_k_dense_replace=0,
            n_group=2,
            topk_group=1,
        )
        model = llada2_moe.Model(args)
        mask_id = 999
        eos_id = 998

        prompt = mx.array([[1, 2, 3, 4, 5]])

        output = llada2_generate.generate(
            model,
            prompt,
            max_new_tokens=16,
            block_length=8,
            steps=8,
            temperature=0.0,
            mask_id=mask_id,
            eos_id=eos_id,
        )

        # Check that no mask tokens appear in the output
        has_mask = mx.any(output == mask_id).item()
        self.assertFalse(has_mask)


class TestLLaDA2MoeWeightSanitization(unittest.TestCase):
    """Tests for weight sanitization."""

    def setUp(self):
        """Create model for testing."""
        self.args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            num_experts=4,
            moe_intermediate_size=32,
            first_k_dense_replace=1,
        )
        self.model = llada2_moe.Model(self.args)

    def test_sanitize_word_embeddings(self):
        """Test word_embeddings to embed_tokens remapping."""
        weights = {
            "model.word_embeddings.weight": mx.zeros((1000, 64)),
        }

        sanitized = self.model.sanitize(weights)

        self.assertIn("model.embed_tokens.weight", sanitized)
        self.assertNotIn("model.word_embeddings.weight", sanitized)

    def test_sanitize_removes_rotary_emb(self):
        """Test that rotary embedding buffers are removed."""
        weights = {
            "model.layers.0.attention.rotary_emb.inv_freq": mx.zeros((32,)),
            "model.layers.0.attention.query_key_value.weight": mx.zeros((192, 64)),
        }

        sanitized = self.model.sanitize(weights)

        self.assertNotIn("model.layers.0.attention.rotary_emb.inv_freq", sanitized)


class TestLLaDA2MoeModelArgs(unittest.TestCase):
    """Tests for model arguments post-initialization."""

    def test_default_kv_heads(self):
        """Test that num_key_value_heads defaults correctly when set to 0."""
        args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=0,  # Should default to num_attention_heads
            vocab_size=1000,
        )

        self.assertEqual(args.num_key_value_heads, 8)

    def test_head_dim_calculation(self):
        """Test head_dim is calculated when not provided."""
        args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
        )

        self.assertEqual(args.head_dim, 16)  # 64 / 4 = 16

    def test_intermediate_size_default(self):
        """Test intermediate_size default calculation."""
        args = llada2_moe.ModelArgs(
            model_type="llada2_moe",
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
            intermediate_size=None,
        )

        self.assertEqual(args.intermediate_size, 256)  # 64 * 4


if __name__ == "__main__":
    unittest.main()
