import math
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Any, List
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope

@dataclass
class ModelArgs(BaseModelArgs):
    # Model identification
    model_type: str
    # Basic model dimensions
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    # Attention configuration
    num_attention_heads: int
    num_key_value_heads: int
    num_query_groups: int
    head_dim: int
    attention_bias: bool
    # Mamba-specific configuration
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_num_groups: int
    mamba_state_dim: int
    mamba_hidden_act: str
    mamba_proj_bias: bool
    # SSM configuration
    ssm_state_size: int
    chunk_size: int
    conv_kernel: int
    n_groups: int
    # Time step configuration
    time_step_rank: int
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    time_step_limit: Tuple[float, float]
    # MLP configuration
    mlp_bias: bool
    mlp_hidden_act: str
    # Normalization
    layer_norm_epsilon: float
    rms_norm_eps: float
    # Bias settings
    use_bias: bool
    use_conv_bias: bool
    # Other settings
    residual_in_fp32: bool
    rescale_prenorm_residual: bool
    tie_word_embeddings: bool
    # Hybrid pattern
    hybrid_override_pattern: str
    # Derived fields (will be set in __post_init__)
    layers_block_type: List[str] = None
    expand: int = None  # Will be calculated

    def __post_init__(self):
        # Set compatibility fields for existing code
        if self.expand is None:
            self.expand = self.intermediate_size // self.hidden_size
        # Parse hybrid pattern into layer types
        if self.hybrid_override_pattern:
            self.layers_block_type = self._parse_hybrid_pattern(self.hybrid_override_pattern)
        else:
            # Default to all mamba if no pattern specified
            self.layers_block_type = ["mamba"] * self.num_hidden_layers

    def _parse_hybrid_pattern(self, pattern: str) -> List[str]:
        """Parse hybrid pattern string into list of layer types.
        M: Mamba2, *: Attention, -: MLP
        """
        layer_types = []
        for char in pattern:
            if char == 'M':
                layer_types.append("mamba")
            elif char == '*':
                layer_types.append("attention")
            elif char == '-':
                layer_types.append("mlp")
        # Skip any other characters (like spaces or separators)
        # Ensure we have the right number of layers
        if len(layer_types) != self.num_hidden_layers:
            print(f"Warning: Pattern length {len(layer_types)} doesn't match num_hidden_layers {self.num_hidden_layers}")
            # Pad or truncate as needed
            if len(layer_types) < self.num_hidden_layers:
                # Pad with mamba layers
                layer_types.extend(["mamba"] * (self.num_hidden_layers - len(layer_types)))
            else:
                # Truncate
                layer_types = layer_types[:self.num_hidden_layers]
        return layer_types


class HybridMambaAttentionCache:
    """Hybrid cache for Mamba + Attention layers."""
    def __init__(self, args: ModelArgs, batch_size: int = 1, dtype=mx.float32):
        self.batch_size = batch_size
        self.offset = 0  # For RoPE
        self.layers_block_type = args.layers_block_type
        self.args = args
        
        # Cache storage
        self.key_cache = [None] * args.num_hidden_layers
        self.value_cache = [None] * args.num_hidden_layers
        self.conv_states = [None] * args.num_hidden_layers
        self.ssm_states = [None] * args.num_hidden_layers
        
        # Initialize only needed caches
        for i, layer_type in enumerate(self.layers_block_type):
            if layer_type == "mamba":
                conv_dim = args.mamba_num_heads * args.mamba_head_dim + 2 * args.mamba_num_groups * args.ssm_state_size
                self.conv_states[i] = mx.zeros((batch_size, conv_dim, args.conv_kernel - 1), dtype=dtype)
                self.ssm_states[i] = mx.zeros((batch_size, args.mamba_num_heads, args.mamba_head_dim, args.ssm_state_size), dtype=dtype)
            elif layer_type == "attention":
                self.key_cache[i] = mx.zeros((batch_size, args.num_key_value_heads, 0, args.head_dim), dtype=dtype)
                self.value_cache[i] = mx.zeros((batch_size, args.num_key_value_heads, 0, args.head_dim), dtype=dtype)

    # Mamba methods
    def update_conv_state(self, layer_idx: int, new_input: mx.array) -> mx.array:
        new_input_reshaped = new_input.squeeze(axis=1)[:, :, None]
        conv_state = self.conv_states[layer_idx]
        if conv_state.shape[-1] > 1:
            updated_state = mx.concatenate([conv_state[:, :, 1:], new_input_reshaped], axis=-1)
        else:
            updated_state = new_input_reshaped
        self.conv_states[layer_idx] = updated_state
        return mx.concatenate([conv_state, new_input_reshaped], axis=-1)

    def update_ssm_state(self, layer_idx: int, new_state: mx.array):
        self.ssm_states[layer_idx] = new_state

    def get_ssm_state(self, layer_idx: int) -> mx.array:
        return self.ssm_states[layer_idx]

    # Attention methods
    def update_and_fetch(self, layer_idx: int, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache for specific layer and return cached keys/values."""
        if self.key_cache[layer_idx].shape[2] == 0:
            self.key_cache[layer_idx] = keys
            self.value_cache[layer_idx] = values
        else:
            self.key_cache[layer_idx] = mx.concatenate([self.key_cache[layer_idx], keys], axis=2)
            self.value_cache[layer_idx] = mx.concatenate([self.value_cache[layer_idx], values], axis=2)
        
        self.offset = self.key_cache[layer_idx].shape[2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        for i, layer_type in enumerate(self.layers_block_type):
            if layer_type == "mamba":
                if self.conv_states[i] is not None:
                    self.conv_states[i] = mx.zeros_like(self.conv_states[i])
                if self.ssm_states[i] is not None:
                    self.ssm_states[i] = mx.zeros_like(self.ssm_states[i])
            elif layer_type == "attention":
                if self.key_cache[i] is not None:
                    b, h, _, d = self.key_cache[i].shape
                    self.key_cache[i] = mx.zeros((b, h, 0, d))
                    self.value_cache[i] = mx.zeros((b, h, 0, d))
        self.offset = 0


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate.astype(mx.float32))
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).astype(input_dtype)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # Use mamba-specific parameters
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_state_dim
        self.conv_kernel_size = args.conv_kernel
        self.head_dim = args.mamba_head_dim
        self.intermediate_size = self.num_heads * self.head_dim
        self.use_conv_bias = args.use_conv_bias
        self.n_groups = args.mamba_num_groups
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups
        self.chunk_size = args.chunk_size
        self.use_bias = args.mamba_proj_bias
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # Match expected weight names from the error
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=args.conv_kernel, groups=self.conv_dim, bias=args.use_conv_bias)
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=self.use_bias)
        self.dt_bias = mx.ones(self.num_heads)
        A = mx.arange(1, self.num_heads + 1, dtype=mx.float32)
        self.A_log = mx.log(A)
        self.D = mx.ones(self.num_heads)
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)
        
        # Pre-compute split indices
        d_mlp = 0
        self.proj_splits = [
            d_mlp,
            2 * d_mlp,
            2 * d_mlp + self.intermediate_size,
            2 * d_mlp + self.intermediate_size + self.conv_dim
        ]
        self.state_splits = [
            self.intermediate_size,
            self.intermediate_size + self.n_groups * self.ssm_state_size
        ]

    @property
    def neg_A(self):
        return -mx.exp(self.A_log.astype(mx.float32))

    def _apply_incremental_conv(self, conv_input: mx.array, cache: HybridMambaAttentionCache) -> mx.array:
        """Apply 1D convolution for incremental inference."""
        conv_window = cache.update_conv_state(self.layer_idx, conv_input)
        # Use the conv1d layer weights
        conv_output = mx.sum(conv_window * self.conv1d.weight.squeeze(), axis=-1)
        if hasattr(self.conv1d, 'bias') and self.conv1d.bias is not None:
            conv_output = conv_output + self.conv1d.bias
        return conv_output[:, None, :]

    def _apply_batch_conv(self, conv_input: mx.array) -> mx.array:
        """Apply 1D convolution for batch processing."""
        # Transpose for conv1d: (batch, channels, length)
        conv_input_t = conv_input.transpose(0, 2, 1)
        conv_output_t = self.conv1d(conv_input_t)
        # Transpose back: (batch, length, channels)
        return conv_output_t.transpose(0, 2, 1)

    def _incremental_ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: HybridMambaAttentionCache
    ) -> mx.array:
        """Optimized SSM for single token generation."""
        batch_size = hidden_states.shape[0]
        dt = nn.softplus(dt.squeeze(axis=1) + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])
        hidden_states = hidden_states.reshape(batch_size, self.num_heads, self.head_dim)
        B = mx.repeat(B.reshape(batch_size, self.n_groups, self.ssm_state_size),
                     self.heads_per_group, axis=1)
        C = mx.repeat(C.reshape(batch_size, self.n_groups, self.ssm_state_size),
                     self.heads_per_group, axis=1)
        dt_expanded = dt[:, :, None, None]
        A_expanded = self.neg_A[None, :, None, None]
        dA = mx.exp(dt_expanded * A_expanded)
        dB = dt_expanded * B[:, :, None, :]
        current_state = cache.get_ssm_state(self.layer_idx)
        new_state = dA * current_state + dB * hidden_states[:, :, :, None]
        cache.update_ssm_state(self.layer_idx, new_state)
        y = mx.sum(C[:, :, None, :] * new_state, axis=-1) + self.D[None, :, None] * hidden_states
        return y.reshape(batch_size, 1, self.intermediate_size)

    def _batch_ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array
    ) -> mx.array:
        """Optimized SSM for batch processing."""
        batch_size, seq_len, _ = hidden_states.shape
        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        B = mx.tile(B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
                   (1, 1, self.heads_per_group, 1))
        C = mx.tile(C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
                   (1, 1, self.heads_per_group, 1))
        outputs = []
        h = mx.zeros((batch_size, self.num_heads, self.head_dim, self.ssm_state_size))
        for t in range(seq_len):
            dt_t = dt[:, t, :, None, None]
            A_t = self.neg_A[None, :, None, None]
            dA = mx.exp(dt_t * A_t)
            dB = dt_t * B[:, t, :, None, :]
            h = dA * h + dB * hidden_states[:, t, :, :, None]
            y_t = mx.sum(C[:, t, :, None, :] * h, axis=-1) + \
                  self.D[None, :, None] * hidden_states[:, t]
            outputs.append(y_t)
        y = mx.stack(outputs, axis=1)
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[HybridMambaAttentionCache] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        is_incremental = cache is not None and seq_len == 1
        projected = self.in_proj(hidden_states)
        splits = mx.split(projected, self.proj_splits, axis=-1)
        _, _, gate, conv_input, dt = splits
        
        if is_incremental:
            conv_output = self._apply_incremental_conv(conv_input, cache)
        else:
            conv_output = self._apply_batch_conv(conv_input)
        
        conv_output = nn.silu(conv_output)
        hidden_states, B, C = mx.split(conv_output, self.state_splits, axis=-1)
        
        if is_incremental:
            y = self._incremental_ssm(hidden_states, B, C, dt, cache)
        else:
            y = self._batch_ssm(hidden_states, B, C, dt)
        
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias)
        # Initialize RoPE
        self.rope = initialize_rope(
            self.head_dim,
            getattr(args, 'rope_theta', 10000.0),
            False,
            getattr(args, 'rope_scaling', None),
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[HybridMambaAttentionCache] = None,
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(self.layer_idx, keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)
        # Handle different activation functions
        if args.mlp_hidden_act == "relu2":
            self.activation = lambda x: nn.relu(x) ** 2
        elif args.mlp_hidden_act == "silu":
            self.activation = nn.silu
        else:
            self.activation = nn.silu  # Default

    def __call__(self, x):
        return self.down_proj(self.activation(self.up_proj(x)))


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.residual_in_fp32 = args.residual_in_fp32
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Get block type from parsed pattern
        self.block_type = args.layers_block_type[layer_idx]
        
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(args, layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(args, layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(args)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

    def __call__(self, x, attention_mask=None, cache: Optional[HybridMambaAttentionCache] = None):
        residual = x
        hidden_states = self.norm(x)
        if self.residual_in_fp32:
            residual = residual.astype(mx.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask, cache=cache)
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states, mask=attention_mask, cache=cache)
        elif self.block_type == "mlp":
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [NemotronHBlock(args, layer_idx=idx) for idx in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _create_causal_mask_4d(self, inputs, cache=None):
        """Create 4D causal mask for attention layers."""
        B, L = inputs.shape[:2]
        
        if cache is not None and hasattr(cache, 'offset'):
            offset = cache.offset
            # Create mask accounting for cache
            mask = mx.full((B, 1, L, L + offset), float('-inf'))
            for i in range(L):
                mask[:, :, i, :offset + i + 1] = 0
        else:
            # Standard 4D causal mask
            mask = mx.full((B, 1, L, L), float('-inf'))
            mask = mx.triu(mask, k=1)
        
        return mask

    def _create_mamba_mask_2d(self, inputs):
        """Create 2D base attention mask for Mamba layers."""
        B, L = inputs.shape[:2]
        # Simple 2D mask - typically all ones for Mamba (no padding assumed)
        # You might need to modify this based on your specific requirements
        mask = mx.ones((B, L), dtype=mx.float32)
        return mask

    def __call__(self,
                 inputs,
                 causal_mask: Optional[mx.array] = None,
                 mamba_mask: Optional[mx.array] = None,
                 cache: Optional[HybridMambaAttentionCache] = None):
        hidden_states = self.embeddings(inputs)
        
        # Create masks if not provided
        if causal_mask is None:
            causal_mask = self._create_causal_mask_4d(hidden_states, cache)  # 4D causal mask
        
        if mamba_mask is None:
            mamba_mask = self._create_mamba_mask_2d(hidden_states)  # 2D base attention mask

        # Process through layers
        for layer_idx, mixer_block in enumerate(self.layers):
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            if mixer_block.block_type == "mamba":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type == "mlp":
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {mixer_block.block_type}")
            
            hidden_states = mixer_block(
                hidden_states,
                attention_mask=layer_mask,
                cache=cache
            )

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        causal_mask: Optional[mx.array] = None,
        mamba_mask: Optional[mx.array] = None,
        cache: Optional[HybridMambaAttentionCache] = None,
    ):
        out = self.backbone(inputs, causal_mask=causal_mask, mamba_mask=mamba_mask, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self, batch_size: int = 1, dtype=mx.float32) -> HybridMambaAttentionCache:
        """Create a properly initialized cache for the hybrid model."""
        return HybridMambaAttentionCache(self.args, batch_size=batch_size, dtype=dtype)

    def sanitize(self, weights):
        """Remove unused weights for the specific architecture."""
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k:
                if len(v.shape) == 3:
                    v = v.transpose(0, 2, 1)
            sanitized[k] = v
        return sanitized