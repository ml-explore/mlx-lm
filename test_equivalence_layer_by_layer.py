
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

def test_norm():
    print("\n--- Testing RMSNorm ---")
    # PT: SarvamMoERMSNorm
    # MLX: SarvamMoERMSNorm
    
    hidden_size = 64
    eps = 1e-6
    
    pt_norm = nn.Module() # Mock container if needed, or just instantiate class
    # SarvamMoE Transformers defines SarvamMoERMSNorm
    from mlx_lm.models.sarvam_moe_transformers import SarvamMoERMSNorm as PTSarvamMoERMSNorm
    from mlx_lm.models.sarvam_moe import SarvamMoERMSNorm
    
    pt_norm = PTSarvamMoERMSNorm(hidden_size, eps=eps).eval()
    mlx_norm = SarvamMoERMSNorm(hidden_size, eps=eps)
    
    copy_weights_norm(mlx_norm, pt_norm)
    
    x_pt = torch.randn(1, 10, hidden_size)
    x_mlx = torch_to_mlx(x_pt)
    
    with torch.no_grad():
        out_pt = pt_norm(x_pt)
    out_mlx = mlx_norm(x_mlx)
    
    check_close(out_mlx, out_pt, name="RMSNorm Output")

    check_close(out_mlx, out_pt, name="RMSNorm Output")

def test_rotary_embedding():
    print("\n--- Testing Rotary Embedding ---")
    # PT: SarvamMoERotaryEmbedding
    # MLX: SarvamMoERotaryEmbedding
    
    hidden_size = 64
    num_heads = 4
    head_dim = 16
    partial_rotary_factor = 1.0 # Test full first
    rope_theta = 10000.0
    
    config = SarvamMoEConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        rope_theta=rope_theta
    )
    
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads, # match
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        rope_theta=rope_theta,
        vocab_size=1000,
        intermediate_size=128,
        num_hidden_layers=1,
    )
    
    # PT
    from mlx_lm.models.sarvam_moe_transformers import SarvamMoERotaryEmbedding as PTSarvamMoERotaryEmbedding
    pt_rope_emb = PTSarvamMoERotaryEmbedding(config)
    
    # MLX
    from mlx_lm.models.sarvam_moe import SarvamMoERotaryEmbedding
    mlx_rope_emb = SarvamMoERotaryEmbedding(args)
    
    # Input
    B, L = 1, 10
    pos_ids = torch.arange(L).unsqueeze(0) # (1, L)
    pos_ids_mlx = torch_to_mlx(pos_ids)
    
    x_dummy = torch.randn(B, L, hidden_size) # for device inference in PT
    x_dummy_mlx = torch_to_mlx(x_dummy)
    
    # Forward
    # PT forward(x, position_ids) -> cos, sin
    with torch.no_grad():
        cos_pt, sin_pt = pt_rope_emb(x_dummy, pos_ids)
        
    cos_mlx, sin_mlx = mlx_rope_emb(x_dummy_mlx, pos_ids_mlx)
    
    # Compare
    check_close(cos_mlx, cos_pt, name="RoPE Cos Output")
    check_close(sin_mlx, sin_pt, name="RoPE Sin Output")

def test_attention():
    print("\n--- Testing Attention ---")
    set_seeds()  # Ensure reproducible results
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
    
    # Create position embeddings using the RoPE classes for consistency
    # PT
    from mlx_lm.models.sarvam_moe_transformers import SarvamMoERotaryEmbedding as PTRoPE
    pt_rope = PTRoPE(config)
    pos_ids_pt = torch.arange(L).unsqueeze(0)
    with torch.no_grad():
        cos_pt, sin_pt = pt_rope(x_pt, pos_ids_pt)
    
    # MLX
    from mlx_lm.models.sarvam_moe import SarvamMoERotaryEmbedding
    mlx_rope = SarvamMoERotaryEmbedding(args)
    pos_ids_mlx = mx.arange(L)[None]
    cos_mlx, sin_mlx = mlx_rope(x_mlx, pos_ids_mlx)
    
    # Forward PT
    with torch.no_grad():
        out_pt, _, _ = pt_attn(x_pt, position_embeddings=(cos_pt, sin_pt))

    # Forward MLX
    out_mlx = mlx_attn(x_mlx, mask=None, position_embeddings=(cos_mlx, sin_mlx))
    
    check_close(out_mlx, out_pt, name="Attention Output")

def test_gate():
    print("\n--- Testing Gate ---")
    # PT: SarvamMoEGate
    # MLX: SarvamMoEGate
    
    hidden_size = 64
    num_experts = 4
    num_experts_per_tok = 2
    n_group = 1
    topk_group = 1
    routed_scaling_factor = 1.0
    
    config = SarvamMoEConfig(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        moe_router_enable_expert_bias=True
    )
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        moe_router_enable_expert_bias=True,
        # dummies
        vocab_size=1000,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128
    )
    
    from mlx_lm.models.sarvam_moe_transformers import SarvamMoEGate as PTSarvamMoEGate
    from mlx_lm.models.sarvam_moe import SarvamMoEGate
    
    pt_gate = PTSarvamMoEGate(config).eval()
    mlx_gate = SarvamMoEGate(args)
    
    # Sync weights
    # PT: weight (NumExperts, Hidden), expert_bias (NumExperts)
    # MLX: weight (NumExperts, Hidden), expert_bias (NumExperts)
    mlx_gate.weight = torch_to_mlx(pt_gate.weight)
    mlx_gate.expert_bias = torch_to_mlx(pt_gate.expert_bias)
    
    x_pt = torch.randn(1, 10, hidden_size)
    x_mlx = torch_to_mlx(x_pt)
    
    with torch.no_grad():
        # returns topk_idx, topk_weight, logits
        pt_inds, pt_weights, _ = pt_gate(x_pt)
        
    # MLX returns inds, topk_weight, logits
    mlx_inds, mlx_weights, mlx_logits = mlx_gate(x_mlx)
    
    check_close(mlx_inds, pt_inds, name="Gate Indices")
    check_close(mlx_weights, pt_weights, name="Gate Weights")
    
    # PT Gate returns logits as well?
    # pt_gate returns (inds, weights, logits)
    with torch.no_grad():
        _, _, pt_logits = pt_gate(x_pt)
        
    check_close(mlx_logits, pt_logits, name="Gate Logits")

    check_close(mlx_inds, pt_inds, name="Gate Indices")
    check_close(mlx_weights, pt_weights, name="Gate Weights")

def test_experts():
    print("\n--- Testing Experts ---")
    config = SarvamMoEConfig(
        hidden_size=64,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        # dummies
        vocab_size=1000,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128
    )
    
    from mlx_lm.models.sarvam_moe_transformers import SarvamMoEExperts as PTSarvamMoEExperts
    from mlx_lm.models.sarvam_moe import SarvamMoEExperts
    
    pt_experts = PTSarvamMoEExperts(config).eval()
    mlx_experts = SarvamMoEExperts(args)
    
    # Sync Weights
    # PT: experts is list of MLPs (gate, up, down)
    # MLX: switch_mlp (gate, up, down - stacked)
    gate_w, up_w, down_w = [], [], []
    for i in range(config.num_experts):
        gate_w.append(pt_experts[i].gate_proj.weight)
        up_w.append(pt_experts[i].up_proj.weight)
        down_w.append(pt_experts[i].down_proj.weight)
    
    mlx_experts.switch_mlp.gate_proj.weight = torch_to_mlx(torch.stack(gate_w))
    mlx_experts.switch_mlp.up_proj.weight = torch_to_mlx(torch.stack(up_w))
    mlx_experts.switch_mlp.down_proj.weight = torch_to_mlx(torch.stack(down_w))
    
    # Inputs
    # Experts take (x, inds, weights)
    # Need dummy inds and weights
    B, L = 1, 10
    topk = config.num_experts_per_tok
    num_ex = config.num_experts
    
    x_pt = torch.randn(B, L, 64)
    x_mlx = torch_to_mlx(x_pt)
    
    
    # Random topk indices (Must be unique per token for Reference Inference Logic)
    # Generate random logits (B, L, NumExperts)
    logits = torch.randn(B, L, num_ex)
    # Get topk indices
    _, inds_pt = torch.topk(logits, topk, dim=-1)
    
    inds_mlx = torch_to_mlx(inds_pt)
    
    # Random weights
    weights_pt = torch.rand(B, L, topk)
    weights_pt = weights_pt / weights_pt.sum(dim=-1, keepdim=True)
    weights_mlx = torch_to_mlx(weights_pt)
    
    # Forward
    # PT forward(hidden_states, top_k_index, top_k_weights)
    # Note: PT forward implementation in reference expects FLATTENED inputs?
    # No, signature says: hidden_states: (tokens, hidden) or (batch*seq, hidden)
    # top_k_index: (tokens, top_k)
    # But SarvamMoESparseMoeBlock line 344 flattens them before calling experts.
    # So we should pass FLATTENED inputs to test `SarvamMoEExperts` directly, or check if it handles unflattened.
    # Reference Line 269: tokens, hidden_dim = hidden_states.shape
    # flat_topk_idx = top_k_index.view(-1)
    # So it expects 2D hidden_states (Tokens, Dim).
    
    # Flatten inputs
    x_pt_flat = x_pt.view(-1, 64)
    inds_pt_flat = inds_pt.view(-1, topk)
    weights_pt_flat = weights_pt.view(-1, topk)
    
    x_mlx_flat = x_mlx.reshape(-1, 64)
    inds_mlx_flat = inds_mlx.reshape(-1, topk)
    weights_mlx_flat = weights_weights_mlx = weights_mlx.reshape(-1, topk)
    
    with torch.no_grad():
        out_pt = pt_experts(x_pt_flat, inds_pt_flat, weights_pt_flat)
        
    out_mlx = mlx_experts(x_mlx_flat, inds_mlx_flat, weights_mlx_flat)
    
    check_close(out_mlx, out_pt, name="Experts Output", atol=1e-2)

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
        out_pt, logits_pt = pt_block(x_pt)
    out_mlx, logits_mlx = mlx_block(x_mlx)
    
    
    check_close(out_mlx, out_pt, name="MoE Block Output", atol=1e-2)
    check_close(logits_mlx, logits_pt[0], name="MoE Block Logits", atol=1e-2)

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
    
    # MLX
    from mlx_lm.models.sarvam_moe import SarvamMoERotaryEmbedding
    mlx_rope = SarvamMoERotaryEmbedding(args)
    pos_ids_mlx = mx.arange(10)[None]
    cos_mlx, sin_mlx = mlx_rope(x_mlx, pos_ids_mlx)
    
    with torch.no_grad():
        pt_outputs = pt_layer(x_pt, position_embeddings=(cos_pt, sin_pt), output_router_logits=True)
        out_pt = pt_outputs[0]
        logits_pt = pt_outputs[-1] 
    
    # MLX DecoderLayer expects mask=None for test?
    # Actually need a mask for cache even if cache is None? No.
    # Pass position_embeddings
    out_mlx, logits_mlx = mlx_layer(x_mlx, mask=None, position_embeddings=(cos_mlx, sin_mlx), output_router_logits=True)
    
    
    check_close(out_mlx, out_pt, name="Decoder Layer Output", atol=1e-2)
    check_close(logits_mlx, logits_pt[0], name="Decoder Layer Logits", atol=1e-2)

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
    
    check_close(out_mlx, out_pt, name="Full Model Output")

    # Check output_router_logits flag
    print("\n--- Checking output_router_logits flag ---")
    from mlx_lm.models.sarvam_moe import SarvamMoEModelOutputWithPast
    
    # PT with output_router_logits=True
    with torch.no_grad():
        pt_out_struct = pt_model(x_pt, output_router_logits=True)
        # pt_out_struct is MoeModelOutputWithPast
        pt_logits = pt_out_struct.router_logits
    
    # MLX with output_router_logits=True
    mlx_out_struct = mlx_model(x_mlx, output_router_logits=True)
    
    if isinstance(mlx_out_struct, SarvamMoEModelOutputWithPast):
        print("✅ Returned correct class SarvamMoEModelOutputWithPast")
    else:
        print(f"❌ Returned wrong type: {type(mlx_out_struct)}")
    
    if mlx_out_struct.router_logits is not None:
        print("✅ router_logits is not None")
        # Check values
        # router_logits is a tuple of logits per layer
        # Check first layer
        # Note: pt_logits tuple length should match num_layers
        if len(mlx_out_struct.router_logits) == len(pt_logits):
             print("✅ router_logits length matches")
             for i, (ml, pl) in enumerate(zip(mlx_out_struct.router_logits, pt_logits)):
                 if ml is not None and pl is not None:
                     check_close(ml, pl[0], name=f"Layer {i} Router Logits", atol=1e-2)
        else:
             print(f"❌ router_logits length mismatch: MLX {len(mlx_out_struct.router_logits)} vs PT {len(pt_logits)}")
    else:
        print("❌ router_logits is None")

def test_full_equivalence():
    set_seeds()
    test_norm()
    test_rotary_embedding()
    test_attention()
    test_gate()
    test_experts()
    test_mlp()
    test_moe_block()
    test_decoder_layer()
    test_model()

if __name__ == "__main__":
    test_full_equivalence()
