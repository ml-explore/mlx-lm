
import mlx.core as mx
from mlx_lm.models.sarvam_moe import SarvamMoEAttention, ModelArgs

def test_sarvam_attention():
    print("Testing SarvamMoEAttention...")
    
    # 1. Setup Args
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        partial_rotary_factor=0.5,
        head_dim=16
    )
    
    # 2. Instantiate
    attn = SarvamMoEAttention(args)
    
    # 3. Create dummy input (B, L, hidden_size)
    B, L, D = 1, 10, 64
    x = mx.random.normal(shape=(B, L, D))
    
    # 4. Forward pass
    try:
        y = attn(x)
        print(f"Forward pass successful. Output shape: {y.shape}")
        
        assert y.shape == (B, L, D), f"Expected {(B, L, D)}, got {y.shape}"
        
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
        raise e
    except Exception as e:
        print(f"Caught unexpected Exception: {e}")
        raise e

    print("Test passed!")

if __name__ == "__main__":
    test_sarvam_attention()
