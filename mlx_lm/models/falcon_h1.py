import inspect
from typing import Optional, Tuple


import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass, field
from typing import List, Optional
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope

@dataclass
class ModelArgs(BaseModelArgs):
    architectures: List[str] = field(default_factory=lambda: ["FalconH1ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.9375
    attn_layer_indices: Optional[List[int]] = None
    bos_token_id: int = 1
    embedding_multiplier: float = 5.656854249492381
    eos_token_id: int = 11
    head_dim: int = 64
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    key_multiplier: float = 0.390625
    lm_head_multiplier: float = 0.0390625
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 64
    mamba_d_ssm: int = 1536
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 24
    mamba_norm_before_gate: bool = False
    mamba_proj_bias: bool = False
    mamba_rms_norm: bool = False
    mamba_use_mlp: bool = True
    max_position_embeddings: int = 131072
    mlp_bias: bool = False
    mlp_expansion_factor: int = 8
    mlp_multipliers: List[float] = field(default_factory=lambda: [0.8838834764831844, 0.5859375])
    model_type: str = "falcon_h1"
    num_attention_heads: int = 8
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    num_logits_to_keep: int = 1
    pad_token_id: int = 0
    projectors_bias: bool = False
    rms_norm_eps: float = 1e-05
    rope_traditional: bool = False
    rope_scaling: Optional[float] = None
    rope_theta: float = 100000000000.0
    ssm_in_multiplier: float = 1.25
    ssm_multipliers: List[float] = field(default_factory=lambda: [0.3535533905932738, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738])
    ssm_out_multiplier: float = 0.23570226039551587
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    vocab_size: int = 32784


# ========================================
# Parallel Scan Implementation for Mamba
# ========================================

def pscan(A, X):
    """
    Parallel scan operation for Mamba SSM.
    Simplified implementation using MLX operations.

    Args:
        A: State transition matrices [B, L, D_state]
        X: Input sequence [B, L, D_state]

    Returns:
        Y: Output sequence [B, L, D_state]
    """
    B, L, D = X.shape

    # For now, use a simplified sequential scan
    # This can be optimized later with a proper parallel scan
    outputs = []
    h = mx.zeros((B, D))  # Initial hidden state

    for t in range(L):
        h = A[:, t, :] * h + X[:, t, :]  # State update
        outputs.append(h)

    # Stack outputs
    Y = mx.stack(outputs, axis=1)  # [B, L, D]

    return Y

# ========================================
# RMSNorm Gated
# ========================================
class FalconH1RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, n_groups=1, norm_before_gate=True):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype

        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * nn.silu(gate.astype(mx.float32))

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, dim = hidden_states.shape
        else:
            batch_size, dim = hidden_states.shape
            seq_len = 1
        hidden_states = hidden_states.astype(mx.float32)

        hidden_states = hidden_states.reshape(batch_size, seq_len, self.n_groups, int(dim // self.n_groups))
        variance = (hidden_states**2).mean(-1, keepdims=True)

        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight.reshape(self.n_groups, int(dim // self.n_groups)) * hidden_states
        hidden_states = hidden_states.reshape(batch_size, seq_len, dim)

        if seq_len == 1:
            hidden_states = hidden_states.squeeze(1)

        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * nn.silu(gate.astype(mx.float32))
        return hidden_states.astype(input_dtype)



# ========================================
# Compute MuP Vector
# ========================================

def compute_mup_vector(config):
    """
    Computes the MuP vector based on model configuration.

    FalconH1 applies different MuP multiplier for each dimension of the hidden states.
    The MuP vector is partitioned into chunks, and each chunk is multiplied with its
    corresponding projected dimension.

    Args:
        config: FalconH1Config object

    Returns:
        mx.array: The computed MuP vector
    """
    # We'll need some values from the config to compute the vector dimensions
    intermediate_size = (
        config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
    )
    groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
    num_heads = config.mamba_n_heads
    zxbcdt_multipliers = config.ssm_multipliers

    vector_shape = 2 * intermediate_size + 2 * groups_time_state_size + num_heads
    mup_vector = mx.ones((1, 1, vector_shape))

    # Apply multipliers to different sections of the vector
    mup_vector[:, :, :intermediate_size] *= zxbcdt_multipliers[0]
    mup_vector[:, :, intermediate_size : 2 * intermediate_size] *= zxbcdt_multipliers[1]
    mup_vector[:, :, 2 * intermediate_size : 2 * intermediate_size + groups_time_state_size] *= zxbcdt_multipliers[2]
    mup_vector[
        :, :, 2 * intermediate_size + groups_time_state_size : 2 * intermediate_size + 2 * groups_time_state_size
    ] *= zxbcdt_multipliers[3]
    mup_vector[:, :, 2 * intermediate_size + 2 * groups_time_state_size :] *= zxbcdt_multipliers[4]
    return mup_vector


# ========================================
# Mamba2 Cache
# ========================================

class Mamba2Cache:
    """
    Arguments:
        config: Mamba2Config
        batch_size: int

    Attributes:
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config.
        n_groups: (`int`):
            Model's number of groups taken from the config - similar to tensor parallel in Transformer.
        state_size: (`int`):
            Model's SSM state size taken from config.
        num_heads: (`int`):
            The number of heads used in the linear attention / SSM.
        head_dim: (`int`):
            The respective dimension of the heads used in the linear attention / SSM.
        intermediate_size: (`int`):
            Model's intermediate_size based on (expand * hidden_dim) from config.
        conv_states: (`dict`):
            A dict of tensors that holds convolutional states.
        ssm_states: (`dict`):
            A dict of tensors that holds ssm states.
    """

    def __init__(
        self,
        config,
        batch_size: int = 1,
    ):
        self.seqlen_offset = 0
        self.has_previous_state = False
        self.conv_kernel_size = config.mamba_d_conv

        self._seen_tokens = 0

        self.intermediate_size = (
            config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
        )

        self.conv_states = {}
        self.ssm_states = {}

        for i in range(config.num_hidden_layers):
            self.conv_states[i] = mx.zeros(
                (batch_size,
                self.intermediate_size + 2 * config.mamba_n_groups * config.mamba_d_state,
                self.conv_kernel_size)
            )
            self.ssm_states[i] = mx.zeros(
                (batch_size,
                config.mamba_n_heads,
                config.mamba_d_head,
                config.mamba_d_state)
            )

        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            self.transformer_layers.append(i)

        self.key_cache: List[mx.array] = []
        self.value_cache: List[mx.array] = []

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = mx.concatenate([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = mx.concatenate([self.value_cache[layer_idx], value_states], axis=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: mx.array,
        cache_position: mx.array,
    ) -> mx.array:
        conv_state = self.conv_states[layer_idx]
        cache_position = mx.clip(cache_position, 0, self.conv_kernel_size - 1)

        conv_state = mx.roll(conv_state, shift=-1, axis=-1)

        if len(cache_position) > 1:
            conv_state = conv_state.at[:, :, :].set(new_conv_state.transpose(0, 2, 1))
        else:
            conv_state = conv_state.at[:, :, -1].set(new_conv_state[:, :, -1])

        self.conv_states[layer_idx] = conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        for i in range(len(self.conv_states)):
            self.conv_states[i] = mx.zeros_like(self.conv_states[i])
            self.ssm_states[i] = mx.zeros_like(self.ssm_states[i])


# ========================================
# Attention Components
# ========================================

class FalconH1Attention(nn.Module):
    """Multi-head attention component"""

    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim ** -0.5

        self.layer_idx = layer_idx
        self.key_multiplier = config.key_multiplier

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # RoPE
        self.rope = initialize_rope(
            self.head_dim,
            config.rope_theta,
            config.rope_traditional,
            config.rope_scaling,
            config.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        # Linear projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        keys = keys * self.key_multiplier

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.seqlen_offset)
            keys = self.rope(keys, offset=cache.seqlen_offset)
            keys, values = cache.update(keys, values, layer_idx=self.layer_idx)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


# ========================================
# Hybrid Mixer Block
# ========================================

def apply_mask_to_padding_states(input_states, attention_mask):
    """Apply attention mask to padding states"""
    if attention_mask is not None:
        # Expand mask to match input dimensions
        mask = mx.expand_dims(attention_mask, axis=-1)
        input_states = input_states * mask
    return input_states


def pad_tensor_by_size(tensor, pad_size):
    """Pad tensor by specified size"""
    if pad_size > 0:
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_size  # Pad sequence dimension
        padding = mx.zeros(pad_shape, dtype=tensor.dtype)
        tensor = mx.concatenate([tensor, padding], axis=1)
    return tensor


def reshape_into_chunks(tensor, pad_size, chunk_size):
    """Reshape tensor into chunks"""
    if pad_size > 0:
        tensor = pad_tensor_by_size(tensor, pad_size)

    batch_size, seq_len = tensor.shape[:2]
    num_chunks = seq_len // chunk_size

    # Reshape to [batch, num_chunks, chunk_size, ...]
    new_shape = [batch_size, num_chunks, chunk_size] + list(tensor.shape[2:])
    return mx.reshape(tensor, new_shape)


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.shape[-1]
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = mx.expand_dims(input_tensor, axis=-1)
    input_tensor = mx.broadcast_to(input_tensor, input_tensor.shape[:-1] + (chunk_size,))

    # 2. create a lower triangular mask with the diagonal set to 0 to zero out elements above diag
    mask = mx.tri(chunk_size, k=-1, dtype=mx.bool_)  # Lower triangular, diagonal=-1
    input_tensor = mx.where(mask, input_tensor, 0)

    # 3. compute actual cumsum
    tensor_segsum = mx.cumsum(input_tensor, axis=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = mx.tri(chunk_size, k=0, dtype=mx.bool_)  # Lower triangular, diagonal=0
    tensor_segsum = mx.where(mask, tensor_segsum, -mx.inf)

    return tensor_segsum


class FalconH1Mixer(nn.Module):
    def __init__(self, config, layer_idx: int, mup_vector: mx.array, batch_size: int = 1):
        super().__init__()
        self.num_heads = config.mamba_n_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = (
            int(config.mamba_expand * self.hidden_size) if config.mamba_d_ssm is None else config.mamba_d_ssm
        )
        self.layer_idx = layer_idx
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias

        self.layer_norm_epsilon = config.rms_norm_eps
        self.groups_time_state_size = config.mamba_n_groups * self.ssm_state_size

        self.n_groups = config.mamba_n_groups
        self.head_dim = config.mamba_d_head
        self.chunk_size = config.mamba_chunk_size

        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.mamba_proj_bias,
        )

        self.dt_bias = mx.ones(self.num_heads)

        A = mx.arange(1, self.num_heads + 1, dtype=mx.float32)
        self.A_log = mx.log(A)

        self.mamba_rms_norm = config.mamba_rms_norm
        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=config.mamba_norm_before_gate,
            )

        self.D = mx.ones(self.num_heads) + 1.0

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.projectors_bias)
        self.use_bias = config.projectors_bias

        self.ssm_in_multiplier = config.ssm_in_multiplier
        self._mup_vector = mup_vector

    def __call__(self, input_states, cache=None, mask=None, cache_position=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        if mask is not None:
            mask = mask[:1, ...] # only take the first token

        input_states = apply_mask_to_padding_states(input_states, mask)

        # Add Multipliers
        input_states = input_states * self.ssm_in_multiplier
        projected_states = self.in_proj(input_states)
        projected_states = projected_states * self._mup_vector

        # Split projected states
        gate = projected_states[..., :self.intermediate_size]
        hidden_states_B_C = projected_states[..., self.intermediate_size:self.intermediate_size + self.conv_dim]
        dt = projected_states[..., self.intermediate_size + self.conv_dim:]

        use_precomputed_states = (
            cache is not None
            and hasattr(cache, 'has_previous_state')
            and cache.has_previous_state
            and seq_len == 1
            and cache.conv_states[self.layer_idx].shape[0] == batch_size
            and cache.ssm_states[self.layer_idx].shape[0] == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            # Roll cache states
            conv_state = mx.roll(cache.conv_states[self.layer_idx], shift=-1, axis=-1)
            conv_state = conv_state.at[:, :, -1].set(hidden_states_B_C[:, 0, :])
            cache.conv_states[self.layer_idx] = conv_state

            # Convolution using matrix multiplication
            hidden_states_B_C = mx.sum(
                conv_state * mx.squeeze(self.conv1d.weight, axis=1), axis=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = nn.silu(hidden_states_B_C)
        else:
            # Init cache
            if cache is not None:
                hidden_states_B_C_transposed = mx.transpose(hidden_states_B_C, [0, 2, 1])
                # Ensure padding size is non-negative
                seq_len_transposed = hidden_states_B_C_transposed.shape[-1]
                pad_size = max(0, self.conv_kernel_size - seq_len_transposed)

                if pad_size > 0:
                    pad_width = [(0, 0), (0, 0), (pad_size, 0)]
                    conv_states = mx.pad(hidden_states_B_C_transposed, pad_width)
                else:
                    conv_states = hidden_states_B_C_transposed

                cache.conv_states[self.layer_idx] = conv_states

            # Apply 1D convolution
            hidden_states_B_C = nn.silu(self.conv1d(hidden_states_B_C))[:, :seq_len, :]

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, mask)

        # Split hidden states
        hidden_states = hidden_states_B_C[..., :self.intermediate_size]
        B = hidden_states_B_C[..., self.intermediate_size:self.intermediate_size + self.n_groups * self.ssm_state_size]
        C = hidden_states_B_C[..., self.intermediate_size + self.n_groups * self.ssm_state_size:]

        # 3. SSM transformation
        A = -mx.exp(self.A_log.astype(mx.float32))  # [num_heads]

        if use_precomputed_states:
            # Single token generation path
            dt = dt[:, 0, :][:, None, ...]
            dt = mx.transpose(dt, [0, 2, 1])
            dt = mx.broadcast_to(dt, (batch_size, dt.shape[1], self.head_dim))

            # Expand dt_bias
            dt_bias = mx.expand_dims(self.dt_bias, axis=-1)
            dt_bias = mx.broadcast_to(dt_bias, (self.dt_bias.shape[0], self.head_dim))

            dt = nn.softplus(dt + dt_bias.astype(dt.dtype))
            dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

            # Expand A
            A = mx.expand_dims(mx.expand_dims(A, axis=-1), axis=-1)
            A = mx.broadcast_to(A, (self.num_heads, self.head_dim, self.ssm_state_size)).astype(mx.float32)

            # Discretize A
            dA = mx.exp(mx.expand_dims(dt, axis=-1) * A)

            # Discretize B
            B = mx.reshape(B, (batch_size, self.n_groups, -1))
            B = mx.expand_dims(B, axis=2)
            B = mx.broadcast_to(B, (batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]))
            B = mx.reshape(B, (batch_size, -1, B.shape[-1]))

            # Compute dB
            dB = mx.expand_dims(dt, axis=-1) * mx.expand_dims(B, axis=2)

            # Reshape hidden states
            hidden_states = mx.reshape(hidden_states, (batch_size, -1, self.head_dim))
            dBx = dB * mx.expand_dims(hidden_states, axis=-1)

            # State calculation
            new_ssm_state = cache.ssm_states[self.layer_idx] * dA + dBx
            cache.ssm_states[self.layer_idx] = new_ssm_state

            # Subsequent output
            C = mx.reshape(C, (batch_size, self.n_groups, -1))
            C = mx.expand_dims(C, axis=2)
            C = mx.broadcast_to(C, (batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]))
            C = mx.reshape(C, (batch_size, -1, C.shape[-1]))

            ssm_states = cache.ssm_states[self.layer_idx].astype(C.dtype)

            # Reshape for batch matrix multiplication
            ssm_states_reshaped = mx.reshape(ssm_states, (batch_size * self.num_heads, self.head_dim, self.ssm_state_size))
            C_reshaped = mx.reshape(C, (batch_size * self.num_heads, self.ssm_state_size, 1))

            # Batch matrix multiplication using @ operator
            y = ssm_states_reshaped @ C_reshaped
            y = mx.reshape(y, (batch_size, self.num_heads, self.head_dim))

            # D skip connection
            D = mx.expand_dims(self.D, axis=-1)
            D = mx.broadcast_to(D, (self.D.shape[0], self.head_dim))
            y = y + hidden_states * D

            # Reshape output
            y = mx.reshape(y, (batch_size, -1))
            y = mx.expand_dims(y, axis=1)
            import pdb; pdb.set_trace()
        else:
            # Full sequence processing path
            dt = nn.softplus(dt + self.dt_bias)
            dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, -1, self.head_dim)).astype(mx.float32)
            B = mx.reshape(B, (batch_size, seq_len, -1, self.ssm_state_size)).astype(mx.float32)
            C = mx.reshape(C, (batch_size, seq_len, -1, self.ssm_state_size)).astype(mx.float32)

            # Repeat B and C for multiple heads
            B = mx.repeat(B, self.num_heads // self.n_groups, axis=2)
            C = mx.repeat(C, self.num_heads // self.n_groups, axis=2)

            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = mx.expand_dims(self.D, axis=-1) * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * mx.expand_dims(dt, axis=-1)
            A = A.astype(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states = reshape_into_chunks(hidden_states, pad_size, self.chunk_size)
            A = reshape_into_chunks(A, pad_size, self.chunk_size)
            B = reshape_into_chunks(B, pad_size, self.chunk_size)
            C = reshape_into_chunks(C, pad_size, self.chunk_size)

            # Permute A for computation
            A = mx.transpose(A, [0, 3, 1, 2])
            A_cumsum = mx.cumsum(A, axis=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            L = mx.exp(segment_sum(A))

            # Contraction of C and B to get G
            C_expanded = mx.expand_dims(C, axis=3)
            B_expanded = mx.expand_dims(B, axis=2)
            G_intermediate = C_expanded * B_expanded
            G = mx.sum(G_intermediate, axis=-1)

            # Compute M
            L_permuted = mx.transpose(L, [0, 2, 3, 4, 1])
            M_intermediate = mx.expand_dims(G, axis=-1) * mx.expand_dims(L_permuted, axis=-1)
            M = mx.sum(M_intermediate, axis=-1)

            # Compute Y_diag
            hidden_states_expanded = mx.expand_dims(hidden_states, axis=2)
            M_expanded = mx.expand_dims(M, axis=-1)
            Y_diag = mx.sum(M_expanded * hidden_states_expanded, axis=3)

            # 2. Compute the state for each intra-chunk
            decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            decay_states_permuted = mx.transpose(decay_states, [0, 2, 3, 1])
            B_decay = B * mx.expand_dims(decay_states_permuted, axis=-1)

            B_decay_expanded = mx.expand_dims(B_decay, axis=-2)
            hidden_states_expanded = mx.expand_dims(hidden_states, axis=-1)
            states = mx.sum(B_decay_expanded * hidden_states_expanded, axis=2)

            # 3. Compute the inter-chunk SSM recurrence
            if use_precomputed_states:
                previous_states = mx.expand_dims(cache.ssm_states[self.layer_idx], axis=1)
            else:
                previous_states = mx.zeros_like(states[:, :1])

            states = mx.concatenate([previous_states, states], axis=1)

            # Pad and compute decay
            A_cumsum_last = A_cumsum[:, :, :, -1]
            pad_width = [(0, 0), (0, 0), (1, 0)]  # Pad last dimension with 1 zero at the beginning
            padded = mx.pad(A_cumsum_last, pad_width)
            decay_chunk = mx.exp(segment_sum(padded))
            decay_chunk = mx.transpose(decay_chunk, [0, 3, 2, 1])

            decay_expanded = mx.expand_dims(mx.expand_dims(decay_chunk, axis=-1), axis=-1)
            states_expanded = mx.expand_dims(states, axis=2)
            new_states = mx.sum(decay_expanded * states_expanded, axis=1)

            states = new_states[:, :-1]
            ssm_state = new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            state_decay_out = mx.exp(A_cumsum)
            C_expanded = mx.expand_dims(C, axis=-2)
            states_expanded = mx.expand_dims(states, axis=2)
            C_times_states = C_expanded * states_expanded

            state_decay_out_permuted = mx.transpose(state_decay_out, [0, 2, 3, 1])
            C_times_states_sum = mx.sum(C_times_states, axis=-1)
            Y_off = C_times_states_sum * mx.expand_dims(state_decay_out_permuted, axis=-1)

            # Add output of intra-chunk and inter-chunk terms
            y = Y_diag + Y_off

            # Reshape output
            y = mx.reshape(y, (batch_size, -1, self.num_heads, self.head_dim))
            y = y + D_residual

            # Remove padding
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = mx.reshape(y, (batch_size, seq_len, -1))

            # Init cache
            if ssm_state is not None and cache is not None:
                cache.ssm_states[self.layer_idx] = ssm_state

        # Apply normalization or activation
        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * nn.silu(gate)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.astype(dtype))
        return contextualized_states


class FalconH1DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, mup_vector: mx.array):
        super().__init__()
        self.feed_forward = FalconH1MLP(config)

        head_dim = config.hidden_size // config.num_attention_heads
        self.channels_attn = config.num_attention_heads * head_dim + 2 * config.num_key_value_heads * head_dim

        self.mamba = FalconH1Mixer(config=config, layer_idx=layer_idx, mup_vector=mup_vector)

        self.self_attn = FalconH1Attention(config, layer_idx)

        self.attention_in_multiplier = config.attention_in_multiplier
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Mamba2Cache,
        mask: mx.array,
        mamba_mask: mx.array,
        cache_position: mx.array,
        **kwargs,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mamba_hidden_states = self.mamba(
            input_states=hidden_states,
            cache=cache,
            mask=mamba_mask,
            cache_position=cache_position,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

        attention_hidden_states = self.self_attn(
            hidden_states * self.attention_in_multiplier,
            mask=mask,
            cache=cache,
        )
        attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

        hidden_states = mamba_hidden_states + attention_hidden_states

        # residual connection after attention
        hidden_states = residual + hidden_states

        # feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ========================================
# MLP Component
# ========================================

class FalconH1MLP(nn.Module):
    """Feed-forward network"""

    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=config.mlp_bias)

        # Add MLP multipliers
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def __call__(self, x):
        y = self.up_proj(x) * nn.silu(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


# ========================================
# Main Model
# ========================================

class FalconH1Model(nn.Module):
    """Falcon-H1 model implementation"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Transformer layers
        mup_vector = compute_mup_vector(config)
        self.layers = [
            FalconH1DecoderLayer(config, layer_idx=layer_idx, mup_vector=mup_vector)
            for layer_idx in range(config.num_hidden_layers)
        ]

        # Final norm
        self.final_layernorm = nn.RMSNorm(
            self.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-5)
        )

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)

        h = h * self.config.embedding_multiplier

        if mask is None:
            mask = create_attention_mask(h, return_array=True)
            mamba_mask = None

        if cache is None:
            cache = [None] * len(self.layers)

        cache_position = mx.arange(h.shape[1], dtype=mx.int32)

        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c, mask=mask, mamba_mask=mamba_mask, cache_position=cache_position)

        return self.final_layernorm(h)


class Model(nn.Module):
    """Falcon-H1 model with language modeling head"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = FalconH1Model(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        hidden_states = self.model(inputs, mask=mask, cache=cache)
        logits = self.lm_head(hidden_states)
        return logits * self.config.lm_head_multiplier

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "conv1d.weight" in name:
                # MLX Conv1d expects [out_channels, in_channels, kernel_size] format
                param = param.transpose(0, 2, 1)
            sanitized_weights[name] = param
        return sanitized_weights

    def make_cache(self):
        return [Mamba2Cache(self.config) for _ in range(self.config.num_hidden_layers)]