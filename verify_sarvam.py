
import sys
import os
sys.path.append("/Users/rachittibrewal/Documents/mlx/mlx-lm")

try:
    from mlx_lm.models import sarvam_moe
    print("Import successful")
    
    args = sarvam_moe.ModelArgs(
        model_type="sarvam_moe",
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128
    )
    model = sarvam_moe.Model(args)
    print("Model instantiated successfully")
    print(model)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
