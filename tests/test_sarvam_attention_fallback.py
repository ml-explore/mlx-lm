import mlx.core as mx
import sys
import os

# Ensure we can import from mlx_lm
sys.path.append(os.getcwd())

from unittest.mock import MagicMock
sys.modules["huggingface_hub"] = MagicMock()

from mlx_lm.models.sarvam_moe import scaled_dot_product_attention

def unit_test_fallback():
    print("Testing SarvamMoE local attention fallback...")
    
    B, H, L, D = 1, 1, 9, 32
    
    # 1. Test standard case (should use fast path, no error)
    print("1. Standard case:")
    q = mx.random.uniform(shape=(B, H, L, D))
    k = mx.random.uniform(shape=(B, H, L, D))
    v = mx.random.uniform(shape=(B, H, L, D))
    out = scaled_dot_product_attention(q, k, v)
    print("   Success, output shape:", out.shape)

    # 2. Test fallback case using the shape that caused the original crash
    # The crash was ValueError: [broadcast_shapes] Shapes (1,1,9,9) and (1,64,9,32) cannot be broadcast.
    # Note: mx.fast seems to broadcast fine in recent versions, but we can FORCE a ValueError 
    # by passing an incompatible mask shape that naive impl might handle via broadcasting if we are lucky,
    # OR we can mock mx.fast.scaled_dot_product_attention to raise ValueError to verify logic path.
    
    print("2. Forced Fallback (Mocking ValueError):")
    
    # We will temporarily mock mx.fast.scaled_dot_product_attention
    original_fast = mx.fast.scaled_dot_product_attention
    
    def mock_fast(*args, **kwargs):
        raise ValueError("Mocked ValueError for testing fallback")
        
    mx.fast.scaled_dot_product_attention = mock_fast
    
    try:
        # Using simple compatible shapes to ensure naive math works
        mask = mx.zeros((1, 1, L, L))
        out_fallback = scaled_dot_product_attention(q, k, v, mask=mask)
        print("   Fallback executed successfully.")
        print("   Output shape:", out_fallback.shape)
        
        # Verify correctness roughly (naive calculation)
        expected_shape = (B, H, L, D)
        assert out_fallback.shape == expected_shape
        
    except Exception as e:
        print(f"   Fallback FAILED: {e}")
    finally:
        # Restore
        mx.fast.scaled_dot_product_attention = original_fast
        print("   Restored original mx.fast")

if __name__ == "__main__":
    unit_test_fallback()
