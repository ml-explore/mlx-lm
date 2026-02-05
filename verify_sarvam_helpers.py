import mlx.core as mx
from mlx_lm.models.sarvam_moe import _get_unpad_data, _expand_mask, _make_causal_mask, ModelArgs, SarvamMoEModel

def test_helpers():
    print("Testing _make_causal_mask...")
    mask = _make_causal_mask((1, 4), mx.float32)
    print(f"Mask shape: {mask.shape}")
    print(mask)
    # Check causal property: Lower triangle should be 0, Upper (excluding diag) should be -1e9
    # mask[0, 0, :, :]
    # [[0, -1e9, -1e9, -1e9],
    #  [0, 0, -1e9, -1e9],
    #  ...
    
    print("\nTesting _expand_mask...")
    m = mx.array([[1, 1, 0, 0]]) # B=1, L=4. Keep first 2.
    expanded = _expand_mask(m, mx.float32)
    print(f"Expanded shape: {expanded.shape}")
    print(expanded)
    # expect 0 for 1s (keep), -1e9 for 0s (mask)
    
    print("\nTesting _get_unpad_data...")
    # mask: [1, 1, 0]
    #       [1, 0, 0]
    m_unpad = mx.array([[1, 1, 0], [1, 0, 0]])
    indices, cu_seqlens, max_seq = _get_unpad_data(m_unpad)
    print(f"Indices: {indices}")
    print(f"Cu_seqlens: {cu_seqlens}") # Expect [0, 2, 3]
    print(f"Max seq: {max_seq}") # Expect 2

    print("\nInstantiating Model...")
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=4
    )
    model = SarvamMoEModel(args)
    # Test forward pass with dummy input to trigger mask creation
    x = mx.random.randint(0, 100, (1, 5))
    out = model(x)
    print("Model forward pass successful.")

if __name__ == "__main__":
    test_helpers()
