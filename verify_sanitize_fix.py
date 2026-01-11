
import mlx.core as mx
from mlx_lm.models.sarvam_moe import Model, ModelArgs, SarvamMoEForCausalLM

def test_sanitize_adds_inv_freq():
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128
    )
    
    # Initialize model
    model = Model(args)
    
    # Check if inv_freq matches logic
    expected_inv_freq = model.model.rotary_emb.inv_freq
    assert expected_inv_freq is not None
    
    # Create incomplete weights (missing inv_freq)
    weights = {
        "model.layers.0.attention.query_key_value.weight": mx.zeros((64, 64)), 
    }
    
    # Run sanitize
    sanitized_weights = model.sanitize(weights)
    
    # Check if inv_freq was added
    if "model.rotary_emb.inv_freq" in sanitized_weights:
        print("Success: model.rotary_emb.inv_freq was added by sanitize.")
        assert mx.array_equal(sanitized_weights["model.rotary_emb.inv_freq"], expected_inv_freq)
    else:
        print("Failure: model.rotary_emb.inv_freq was NOT added.")

if __name__ == "__main__":
    test_sanitize_adds_inv_freq()
