
import os
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.sarvam_moe import SarvamMoEAttention, SarvamMoEMLP, SarvamMoESparseMoeBlock, SarvamMoEDecoderLayer, SarvamMoEModel, ModelArgs
from mlx_lm.models.sarvam_moe_transformers import SarvamMoEConfig, SarvamMoEAttention as PTSarvamMoEAttention, SarvamMoEMLP as PTSarvamMoEMLP, SarvamMoESparseMoeBlock as PTSarvamMoESparseMoeBlock, SarvamMoEDecoderLayer as PTSarvamMoEDecoderLayer, SarvamMoEModel as PTSarvamMoEModel

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)

def torch_to_mlx(tensor):
    return mx.array(tensor.detach().numpy())

def mlx_to_torch(array):
    return torch.from_numpy(np.array(array))

def check_close(a, b, atol=1e-3, rtol=1e-3, name="Tensor"):
    if isinstance(a, mx.array):
        a = np.array(a)
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    
    if np.allclose(a, b, atol=atol, rtol=rtol):
        print(f"✅ {name} matches!")
        return True
    else:
        diff = np.abs(a - b).max()
        print(f"❌ {name} mismatch! Max diff: {diff}")
        print(f"  MLX shape: {a.shape}")
        print(f"  PT shape:  {b.shape}")
        return False

def copy_weights_linear(mlx_layer, pt_layer):
    # PyTorch Linear weights are (out, in), MLX Linear weights are (out, in) but used as (in, out) in forward?
    # MLX: x @ W.T + b -> W is (out, in).
    # Correct. Just direct copy of weight and bias.
    if hasattr(pt_layer, "weight") and pt_layer.weight is not None:
         mlx_layer.weight = torch_to_mlx(pt_layer.weight)
    if hasattr(pt_layer, "bias") and pt_layer.bias is not None:
         mlx_layer.bias = torch_to_mlx(pt_layer.bias)

def copy_weights_norm(mlx_layer, pt_layer):
    if hasattr(pt_layer, "weight") and pt_layer.weight is not None:
        mlx_layer.weight = torch_to_mlx(pt_layer.weight)

def test_attention():
    print("\n--- Testing Attention ---")
    config = SarvamMoEConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        attention_dropout=0.0
    )
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        vocab_size=1000,
        intermediate_size=128,
        num_hidden_layers=1,
        num_experts=4, # dummy
        num_experts_per_tok=2,
    )
    
    # Instantiate
    pt_attn = PTSarvamMoEAttention(config, layer_idx=0).eval()
    mlx_attn = SarvamMoEAttention(args)
    
    # Sync Weights
    copy_weights_linear(mlx_attn.query_key_value, pt_attn.query_key_value)
    copy_weights_linear(mlx_attn.dense, pt_attn.dense)
    if config.use_qk_norm:
        copy_weights_norm(mlx_attn.query_layernorm, pt_attn.query_layernorm)
        copy_weights_norm(mlx_attn.key_layernorm, pt_attn.key_layernorm)

    # Input
    B, L, D = 1, 10, 64
    x_pt = torch.randn(B, L, D)
    x_mlx = torch_to_mlx(x_pt)
    
    # Position IDs (transformers uses them for RoPE sometimes, but SarvamMoE uses rotary_emb from model)
    # The reference Attention .forward() accepts position_embeddings.
    # In full model, rotary_emb is calculated outside.
    # Here we need to mock it.
    
    # Create cos, sin
    # PT: (1, 1, L, D_rope)
    rope_dim = int(config.head_dim * config.partial_rotary_factor)
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(L).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Create cos, sin matching SarvamMoE Transformers expectation
    # SarvamMoEAttention in Transformers expects position_embeddings=(cos, sin)
    # apply_rotary_pos_emb does:
    # cos = cos.unsqueeze(1) -> (1, 1, L, D_rope) -> unsqueezed becomes (1, 1, 1, L, D_rope)??
    # Wait, apply_rotary_pos_emb signature: (q, k, cos, sin, unsqueeze_dim=1)
    # q is (B, H, L, D). 
    # If cos is (1, L, D), unsqueeze(1) -> (1, 1, L, D). This broadcasts.
    # 
    # In my script I did: cos_pt = emb.cos()[None, None, :, :] -> (1, 1, L, D_rope)
    # inside apply_rotary_pos_emb: cos.unsqueeze(1) -> (1, 1, 1, L, D_rope). Rank 5.
    # q is (B, H, L, D) -> Rank 4.
    # q_rot * cos -> Rank 5 * Rank 4 -> Broadcasts to Rank 5. Result Rank 5.
    # q_pass is Rank 4.
    # cat([Rank 5, Rank 4]) -> Error.
    
    # Correction: The reference code's `apply_rotary_pos_emb` automatically unsqueezes at dim 1.
    # So we should pass `cos` and `sin` WITHOUT that dimension if we rely on that, OR pass it such that unsqueeze is correct.
    # The reference `SarvamMoERotaryEmbedding` returns `cos` as `(B, L, D)`.
    # Let's match correct shape: (1, L, D_rope).
    
    cos_pt = emb.cos()[None, :, :] # (1, L, D_rope)
    sin_pt = emb.sin()[None, :, :] # (1, L, D_rope)
    
    # MLX RoPE calculates internally if cache is None? No, self.rope(queries)
    # MLX RoPE is standard.
    # We rely on internal calculation. We assume MLX RoPE implementation matches standard.
    
    # Forward PT
    with torch.no_grad():
        # PT expects position_embeddings=(cos, sin)
        out_pt, _, _ = pt_attn(x_pt, position_embeddings=(cos_pt, sin_pt))

    # Forward MLX
    out_mlx = mlx_attn(x_mlx, mask=None) # Mask is None for unmasked attention
    
    check_close(out_mlx, out_pt, name="Attention Output")

def test_mlp():
    print("\n--- Testing MLP ---")
    # Dense MLP
    config = SarvamMoEConfig(
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu"
    )
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        intermediate_size=128,
        vocab_size=1000,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2
    )
    
    pt_mlp = PTSarvamMoEMLP(config, intermediate_size=128).eval()
    mlx_mlp = SarvamMoEMLP(args, intermediate_size=128)
    
    copy_weights_linear(mlx_mlp.gate_proj, pt_mlp.gate_proj)
    copy_weights_linear(mlx_mlp.up_proj, pt_mlp.up_proj)
    copy_weights_linear(mlx_mlp.down_proj, pt_mlp.down_proj)

    x_pt = torch.randn(1, 10, 64)
    x_mlx = torch_to_mlx(x_pt)
    
    with torch.no_grad():
        out_pt = pt_mlp(x_pt)
    out_mlx = mlx_mlp(x_mlx)
    
    check_close(out_mlx, out_pt, name="MLP Output")

def test_moe_block():
    print("\n--- Testing MoE Block ---")
    config = SarvamMoEConfig(
        hidden_size=64,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0, # simpler
        moe_router_enable_expert_bias=True,
        num_shared_experts=1
    )
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        moe_router_enable_expert_bias=True,
        num_shared_experts=1,
        vocab_size=1000,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2
    )
    
    pt_block = PTSarvamMoESparseMoeBlock(config).eval()
    mlx_block = SarvamMoESparseMoeBlock(args)
    
    # Sync Gate
    # PT: self.gate.weight: (E, H)
    # MLX: self.gate.weight: (E, H)
    mlx_block.gate.weight = torch_to_mlx(pt_block.gate.weight)
    if config.moe_router_enable_expert_bias:
        mlx_block.gate.expert_bias = torch_to_mlx(pt_block.gate.expert_bias)
    
    # Sync Experts
    # PT: experts is ModuleList of MLPs
    # MLX: experts is SarvamMoEExperts -> SwitchGLU
    # SwitchGLU expects weights: gate [E, F, I], up [E, F, I], down [E, I, F]?
    # Let's check SarvamMoEExperts implementation.
    # It stacks weights.
    
    gate_w = []
    up_w = []
    down_w = []
    for i in range(config.num_experts):
        gate_w.append(pt_block.experts[i].gate_proj.weight)
        up_w.append(pt_block.experts[i].up_proj.weight)
        down_w.append(pt_block.experts[i].down_proj.weight)
        
    gate_stack = torch.stack(gate_w)
    up_stack = torch.stack(up_w)
    down_stack = torch.stack(down_w)
    
    mlx_block.experts.switch_mlp.gate_proj.weight = torch_to_mlx(gate_stack)
    mlx_block.experts.switch_mlp.up_proj.weight = torch_to_mlx(up_stack)
    mlx_block.experts.switch_mlp.down_proj.weight = torch_to_mlx(down_stack)

    # Sync Shared Experts
    if pt_block.shared_experts:
        copy_weights_linear(mlx_block.shared_experts.gate_proj, pt_block.shared_experts.gate_proj)
        copy_weights_linear(mlx_block.shared_experts.up_proj, pt_block.shared_experts.up_proj)
        copy_weights_linear(mlx_block.shared_experts.down_proj, pt_block.shared_experts.down_proj)

    x_pt = torch.randn(1, 5, 64)
    x_mlx = torch_to_mlx(x_pt)

    with torch.no_grad():
        out_pt, _ = pt_block(x_pt)
    out_mlx = mlx_block(x_mlx)
    
    
    check_close(out_mlx, out_pt, name="MoE Block Output", atol=1e-2)

def test_decoder_layer():
    print("\n--- Testing Decoder Layer ---")
    config = SarvamMoEConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=1.0,
        first_k_dense_replace=0, # All layers MoE
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        moe_router_enable_expert_bias=True,
        attn_implementation="eager",
    )
    config._attn_implementation = "eager"
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=1.0,
        first_k_dense_replace=0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        moe_router_enable_expert_bias=True,
        vocab_size=1000,
        num_hidden_layers=1,
    )
    
    pt_layer = PTSarvamMoEDecoderLayer(config, layer_idx=0).eval()
    mlx_layer = SarvamMoEDecoderLayer(args, layer_idx=0)
    
    # Sync Norms
    copy_weights_norm(mlx_layer.input_layernorm, pt_layer.input_layernorm)
    copy_weights_norm(mlx_layer.post_attention_layernorm, pt_layer.post_attention_layernorm)
    
    # Sync Attention
    copy_weights_linear(mlx_layer.attention.query_key_value, pt_layer.attention.query_key_value)
    copy_weights_linear(mlx_layer.attention.dense, pt_layer.attention.dense)
    if config.use_qk_norm:
        copy_weights_norm(mlx_layer.attention.query_layernorm, pt_layer.attention.query_layernorm)
        copy_weights_norm(mlx_layer.attention.key_layernorm, pt_layer.attention.key_layernorm)
        
    # Sync MLP/MoE
    # Logic similar to MoE block sync
    mlx_mlp = mlx_layer.mlp
    pt_mlp = pt_layer.mlp
    
    if isinstance(mlx_mlp, SarvamMoESparseMoeBlock):
        mlx_mlp.gate.weight = torch_to_mlx(pt_mlp.gate.weight)
        if config.moe_router_enable_expert_bias:
            mlx_mlp.gate.expert_bias = torch_to_mlx(pt_mlp.gate.expert_bias)
            
        gate_w, up_w, down_w = [], [], []
        for i in range(config.num_experts):
            gate_w.append(pt_mlp.experts[i].gate_proj.weight)
            up_w.append(pt_mlp.experts[i].up_proj.weight)
            down_w.append(pt_mlp.experts[i].down_proj.weight)
        
        mlx_mlp.experts.switch_mlp.gate_proj.weight = torch_to_mlx(torch.stack(gate_w))
        mlx_mlp.experts.switch_mlp.up_proj.weight = torch_to_mlx(torch.stack(up_w))
        mlx_mlp.experts.switch_mlp.down_proj.weight = torch_to_mlx(torch.stack(down_w))
        
        if pt_mlp.shared_experts:
            copy_weights_linear(mlx_mlp.shared_experts.gate_proj, pt_mlp.shared_experts.gate_proj)
            copy_weights_linear(mlx_mlp.shared_experts.up_proj, pt_mlp.shared_experts.up_proj)
            copy_weights_linear(mlx_mlp.shared_experts.down_proj, pt_mlp.shared_experts.down_proj)

    # Input
    x_pt = torch.randn(1, 10, 64)
    x_mlx = torch_to_mlx(x_pt)
    
    # RoPE embeddings
    rope_dim = int(config.head_dim * config.partial_rotary_factor)
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(10).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_pt = emb.cos()[None, :, :]
    sin_pt = emb.sin()[None, :, :]
    
    with torch.no_grad():
        out_pt = pt_layer(x_pt, position_embeddings=(cos_pt, sin_pt))[0] # returns tuple
    
    # MLX DecoderLayer expects mask=None for test?
    # Actually need a mask for cache even if cache is None? No.
    out_mlx = mlx_layer(x_mlx, mask=None)
    
    
    check_close(out_mlx, out_pt, name="Decoder Layer Output", atol=1e-2)

def test_model():
    print("\n--- Testing Full SarvamMoEModel ---")
    config = SarvamMoEConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=1.0,
        first_k_dense_replace=0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        moe_router_enable_expert_bias=True,
        attn_implementation="eager",
        vocab_size=1000,
        num_hidden_layers=2, # Test 2 layers
    )
    config._attn_implementation = "eager"
    
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=1.0,
        first_k_dense_replace=0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        moe_router_enable_expert_bias=True,
        vocab_size=1000,
        num_hidden_layers=2,
    )
    
    pt_model = PTSarvamMoEModel(config).eval()
    mlx_model = SarvamMoEModel(args)
    
    # Sync Embeddings
    # PT: word_embeddings (Embedding)
    # MLX: embed_tokens (Embedding)
    # Pytorch Embedding weight is (Num, Dim)
    # MLX Embedding weight is (Num, Dim)
    mlx_model.embed_tokens.weight = torch_to_mlx(pt_model.word_embeddings.weight)
    
    # Sync Layers
    for i in range(config.num_hidden_layers):
        pt_layer = pt_model.layers[i]
        mlx_layer = mlx_model.layers[i]
        
        # Sync Norms
        copy_weights_norm(mlx_layer.input_layernorm, pt_layer.input_layernorm)
        copy_weights_norm(mlx_layer.post_attention_layernorm, pt_layer.post_attention_layernorm)
        
        # Sync Attention
        copy_weights_linear(mlx_layer.attention.query_key_value, pt_layer.attention.query_key_value)
        copy_weights_linear(mlx_layer.attention.dense, pt_layer.attention.dense)
        if config.use_qk_norm:
            copy_weights_norm(mlx_layer.attention.query_layernorm, pt_layer.attention.query_layernorm)
            copy_weights_norm(mlx_layer.attention.key_layernorm, pt_layer.attention.key_layernorm)
            
        # Sync MLP/MoE
        mlx_mlp = mlx_layer.mlp
        pt_mlp = pt_layer.mlp
        
        if isinstance(mlx_mlp, SarvamMoESparseMoeBlock):
            mlx_mlp.gate.weight = torch_to_mlx(pt_mlp.gate.weight)
            if config.moe_router_enable_expert_bias:
                mlx_mlp.gate.expert_bias = torch_to_mlx(pt_mlp.gate.expert_bias)
                
            gate_w, up_w, down_w = [], [], []
            for e in range(config.num_experts):
                gate_w.append(pt_mlp.experts[e].gate_proj.weight)
                up_w.append(pt_mlp.experts[e].up_proj.weight)
                down_w.append(pt_mlp.experts[e].down_proj.weight)
            
            mlx_mlp.experts.switch_mlp.gate_proj.weight = torch_to_mlx(torch.stack(gate_w))
            mlx_mlp.experts.switch_mlp.up_proj.weight = torch_to_mlx(torch.stack(up_w))
            mlx_mlp.experts.switch_mlp.down_proj.weight = torch_to_mlx(torch.stack(down_w))
            
            if pt_mlp.shared_experts:
                copy_weights_linear(mlx_mlp.shared_experts.gate_proj, pt_mlp.shared_experts.gate_proj)
                copy_weights_linear(mlx_mlp.shared_experts.up_proj, pt_mlp.shared_experts.up_proj)
                copy_weights_linear(mlx_mlp.shared_experts.down_proj, pt_mlp.shared_experts.down_proj)

    # Sync Final Norm
    copy_weights_norm(mlx_model.norm, pt_model.norm)

    # Input (Token IDs)
    B, L = 1, 10
    x_pt = torch.randint(0, config.vocab_size, (B, L))
    x_mlx = torch_to_mlx(x_pt)
    
    # Forward
    with torch.no_grad():
        # PT Model returns (last_hidden_state, ...)
        # It handles RoPE internally
        out_pt = pt_model(x_pt).last_hidden_state
    
    out_mlx = mlx_model(x_mlx)
    
    check_close(out_mlx, out_pt, name="Full Model Output", atol=1e-2)

def test_full_equivalence():
    set_seeds()
    test_attention()
    test_mlp()
    test_moe_block()
    test_decoder_layer()
    test_model()

if __name__ == "__main__":
    test_full_equivalence()
