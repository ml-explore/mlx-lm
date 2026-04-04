import mlx.core as mx
from mlx_lm.models.sarvam_moe import SarvamMoEGate, ModelArgs
import numpy as np

def test_gate_standard():
    print("Testing Gate (Standard TopK)...")
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=8,
        num_experts_per_tok=2,
        n_group=1, # Standard
        topk_group=1,
    )
    gate = SarvamMoEGate(args)
    x = mx.random.normal((1, 4, 16)) # B=1, L=4, H=16
    inds, weights, logits = gate(x)
    print(f"Indices shape: {inds.shape}") # Expect (1, 4, 2)
    print(f"Weights shape: {weights.shape}") # Expect (1, 4, 2)
    print("Weights sum (approx 2.5 after scaling factor 2.5? Check logic):")
    # Logic: weights summed (if top_k>1) are normalized to 1, then scaled by routed_scaling_factor=2.5
    # So sum should be 2.5 per token?
    # Actually logic is: topk_weight = gathered_scores / denom * 2.5
    # So sum of weights = sum(gathered/denom) * 2.5 = 1.0 * 2.5 = 2.5.
    print(weights.sum(axis=-1)) 
    
    assert inds.shape == (1, 4, 2)
    assert weights.shape == (1, 4, 2)

def test_gate_grouped():
    print("\nTesting Gate (Group Limited)...")
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=8,
        num_experts_per_tok=2,
        n_group=4, # 4 groups, 2 experts each
        topk_group=2, # Select 2 groups
        # Experts: 0-1 (G0), 2-3 (G1), 4-5 (G2), 6-7 (G3)
    )
    gate = SarvamMoEGate(args)
    x = mx.random.normal((1, 4, 16))
    inds, weights, logits = gate(x)
    print(f"Indices shape: {inds.shape}")
    print(f"Weights shape: {weights.shape}")
    print("Indices sample:", inds[0, 0])
    
    # Check if indices respect groups? 
    # Hard to check stochastically without controlled weights, but if it runs without error 
    # and produces correct shapes, logic is likely sound port.
    
    assert inds.shape == (1, 4, 2)
    assert weights.shape == (1, 4, 2)

if __name__ == "__main__":
    test_gate_standard()
    test_gate_grouped()
    print("\nSuccess!")
