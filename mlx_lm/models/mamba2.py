import math
from dataclasses import dataclass
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import Mamba2Cache

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int
    n_groups: int
    use_bias: bool
    use_conv_bias: bool
    initializer_range: float
    residual_in_fp32: bool
    chunk_size: int
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float]
    time_step_rank: Union[int, str]
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    norm_before_gate: bool = True
    rms_norm: bool = True

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        if gate is not None:
            # Apply SiLU activation to gate
            gate = gate.astype(mx.float32)
            silu_gate = gate * mx.sigmoid(gate)
            hidden_states = hidden_states * silu_gate
        # Compute variance along last dimension
        variance = mx.mean(mx.power(hidden_states, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        # Cast back to original dtype and apply weight
        return self.weight * hidden_states.astype(input_dtype)


def segsum(input_tensor):
    chunk_size = input_tensor.shape[-1]
    input_tensor = mx.expand_dims(input_tensor, -1)
    input_tensor = mx.broadcast_to(input_tensor, (*input_tensor.shape[:-1], chunk_size, chunk_size))
    mask = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), diagonal=-1)
    input_tensor = input_tensor * mask
    tensor_segsum = mx.cumsum(input_tensor, axis=-2)
    mask = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), diagonal=0)
    return mx.where(mask, tensor_segsum, mx.full_like(tensor_segsum, -float('inf')))


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = mx.array(hidden_states * attention_mask[:, :, None], dtype=dtype)
    return hidden_states


class Mamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = int(args.expand * self.hidden_size)
        self.time_step_rank = int(args.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = args.use_conv_bias

        self.layer_norm_epsilon = args.layer_norm_epsilon
        self.rms_norm = args.rms_norm

        self.n_groups = args.n_groups
        self.head_dim = args.head_dim
        self.chunk_size = args.chunk_size

        self.time_step_limit = args.time_step_limit
        self.time_step_min = args.time_step_min
        self.time_step_max = args.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            groups=self.conv_dim,
            padding=args.conv_kernel - 1
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.use_bias,
        )
        self.dt_bias = mx.zeros(self.num_heads) + 1.0
        A = mx.arange(1, self.num_heads + 1)
        self.A_log = mx.log(A)
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = mx.zeros(self.num_heads) + 1.0
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)
        self.use_bias = args.use_bias

    def __call__(self, input_states, cache_params=None, attention_mask=None):
        batch_size, seq_len, _ = input_states.shape

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        
        # Split the projected states
        parts = []
        start_idx = 0
        for size in [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]:
            parts.append(projected_states[..., start_idx:start_idx+size])
            start_idx += size
        _, _, gate, hidden_states_B_C, dt = parts
        
        # 2. Convolution sequence transformation
        cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)
        
        conv_states = cache_params.conv_states[self.layer_idx]
        
        weight = self.conv1d.weight
        weight_reshaped = mx.reshape(weight, (self.conv_dim, -1))
        hidden_states_B_C = mx.sum(conv_states * weight_reshaped, axis=-1)
        
        if self.use_conv_bias:
            hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
        
        hidden_states_B_C = nn.silu(hidden_states_B_C)
        
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        
        # Split hidden_states_B_C
        hidden_states = hidden_states_B_C[..., :self.intermediate_size]
        B = hidden_states_B_C[..., self.intermediate_size:self.intermediate_size + self.n_groups * self.ssm_state_size]
        C = hidden_states_B_C[..., self.intermediate_size + self.n_groups * self.ssm_state_size:]
        
        # 3. SSM transformation
        A = -mx.exp(self.A_log)  # [num_heads]
        
        # Handle dt
        dt = dt[:, 0, :][:, None, ...]
        dt = mx.transpose(dt, (0, 2, 1))  # equivalent to dt.transpose(1, 2)
        dt = mx.broadcast_to(dt, (batch_size, dt.shape[1], self.head_dim))
        
        # Expand dt_bias
        dt_bias = mx.broadcast_to(self.dt_bias[:, None], (self.dt_bias.shape[0], self.head_dim))
        
        dt = nn.softplus(dt + dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])
        
        # Expand A
        A = mx.broadcast_to(A[:, None, None], (self.num_heads, self.head_dim, self.ssm_state_size))
        
        # Calculate dA
        dA = mx.exp(dt[..., None] * A)
        
        # Discretize B
        B = mx.reshape(B, (batch_size, self.n_groups, -1))[..., None, :]
        B = mx.broadcast_to(B, (batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]))
        B = mx.reshape(B, (batch_size, -1, B.shape[-1]))
        
        # Calculate dB
        dB = dt[..., None] * B[..., None, :]
        
        # Discretize x into dB
        hidden_states = mx.reshape(hidden_states, (batch_size, -1, self.head_dim))
        dBx = dB * hidden_states[..., None]
        
        # State calculation
        cache_params.update_ssm_state(
            layer_idx=self.layer_idx,
            new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx
        )
        
        # Subsequent output
        C = mx.reshape(C, (batch_size, self.n_groups, -1))[..., None, :]
        C = mx.broadcast_to(C, (batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]))
        C = mx.reshape(C, (batch_size, -1, C.shape[-1]))
        
        ssm_states = cache_params.ssm_states[self.layer_idx]
        
        # Reshape for batched matrix multiplication
        ssm_states_reshaped = mx.reshape(ssm_states, (batch_size * self.num_heads, self.head_dim, self.ssm_state_size))
        C_reshaped = mx.reshape(C, (batch_size * self.num_heads, self.ssm_state_size, 1))
        
        # Batched matrix multiplication
        y = mx.matmul(ssm_states_reshaped, C_reshaped)
        y = mx.reshape(y, (batch_size, self.num_heads, self.head_dim))
        
        # D skip connection
        D = mx.broadcast_to(self.D[:, None], (self.D.shape[0], self.head_dim))
        y = y + hidden_states * D
        
        # Reshape y
        y = mx.reshape(y, (batch_size, -1))[:, None, ...]
        
        # Apply normalization
        scan_output = self.norm(y, gate)
        
        # 4. Final linear projection
        return self.out_proj(scan_output)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.mixer = Mamba2Mixer(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        residual = x
        normed = self.norm(x)
        if self.residual_in_fp32:
            residual = residual.astype(mx.float32)
        output = self.mixer(normed, cache)
        return output + residual


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args, indx) for indx in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x: mx.array, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)
        
        hidden = x
        for layer, c in zip(self.layers, cache):
            hidden = layer(hidden, c)
        return self.norm_f(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        hidden = self.backbone(inputs, cache)
        
        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        
        return logits

    def make_cache(self):
        return [Mamba2Cache(self.args) for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers