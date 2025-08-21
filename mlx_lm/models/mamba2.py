import math
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Any
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
    ssm_state_size: int = None
    norm_before_gate: bool = True
    max_position_embeddings: int = 2056

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.ssm_state_size is None:
            self.ssm_state_size = self.state_size


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


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.num_heads * args.head_dim
        self.use_conv_bias = args.use_conv_bias
        self.n_groups = args.n_groups
        self.head_dim = args.head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups
        self.chunk_size = args.chunk_size
        self.use_bias = args.use_bias
        
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        
        # Learnable parameters
        self.conv_weight = mx.random.normal((self.conv_dim, args.conv_kernel)) * 0.1
        if args.use_conv_bias:
            self.conv_bias = mx.zeros(self.conv_dim)
        
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.use_bias)
        
        self.dt_bias = mx.ones(self.num_heads)
        A = mx.arange(1, self.num_heads + 1, dtype=mx.float32)
        self.A_log = mx.log(A)
        self.D = mx.ones(self.num_heads)
        
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)
        
        # Pre-compute split indices - fix for d_mlp
        d_mlp = 0  # Based on the original code
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

    def _apply_incremental_conv(self, conv_input: mx.array, cache: Mamba2Cache) -> mx.array:
        """Apply 1D convolution for incremental inference."""
        # Get the full convolution window (past states + current input)
        conv_window = cache.update_conv_state(self.layer_idx, conv_input)
        # conv_window shape: (batch, conv_dim, conv_kernel_size)
        
        # Apply convolution: sum over the kernel dimension
        conv_output = mx.sum(conv_window * self.conv_weight[None, :, :], axis=-1)
        # conv_output shape: (batch, conv_dim)
        
        # Add bias if present
        if self.use_conv_bias:
            conv_output = conv_output + self.conv_bias
            
        # Reshape to match expected output: (batch, 1, conv_dim)
        return conv_output[:, None, :]

    def _apply_batch_conv(self, conv_input: mx.array) -> mx.array:
        """Apply 1D convolution for batch processing."""
        batch_size, seq_len, conv_dim = conv_input.shape
        
        # Pad the input for causal convolution
        padded_input = mx.pad(conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])
        
        # Apply convolution using sliding windows
        outputs = []
        for i in range(seq_len):
            # Get the conv window for position i
            window = padded_input[:, i:i + self.conv_kernel_size, :]  # (batch, kernel_size, conv_dim)
            # Transpose to (batch, conv_dim, kernel_size) to match self.conv_weight shape (conv_dim, kernel_size)
            window = mx.transpose(window, (0, 2, 1))
            # Apply convolution: elementwise multiply then sum over kernel dimension -> (batch, conv_dim)
            conv_out = mx.sum(window * self.conv_weight[None, :, :], axis=-1)
            outputs.append(conv_out)
        
        conv_output = mx.stack(outputs, axis=1)  # (batch, seq_len, conv_dim)
        
        if self.use_conv_bias:
            conv_output = conv_output + self.conv_bias
            
        return conv_output

    def _incremental_ssm(
            self,
            hidden_states: mx.array,
            B: mx.array,
            C: mx.array, 
            dt: mx.array,
            cache: Mamba2Cache
        ) -> mx.array:
        """Optimized SSM for single token generation."""
        batch_size = hidden_states.shape[0]
        
        # Efficient reshaping without intermediate steps
        dt = nn.softplus(dt.squeeze(axis=1) + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])
        
        # Direct reshape to target dimensions
        hidden_states = hidden_states.reshape(batch_size, self.num_heads, self.head_dim)
        B = mx.repeat(B.reshape(batch_size, self.n_groups, self.ssm_state_size), 
                     self.heads_per_group, axis=1)
        C = mx.repeat(C.reshape(batch_size, self.n_groups, self.ssm_state_size), 
                     self.heads_per_group, axis=1)
        
        # Vectorized discretization
        dt_expanded = dt[:, :, None, None]
        A_expanded = self.neg_A[None, :, None, None]
        dA = mx.exp(dt_expanded * A_expanded)
        dB = dt_expanded * B[:, :, None, :]
        
        # Efficient state update
        current_state = cache.get_ssm_state(self.layer_idx)
        new_state = dA * current_state + dB * hidden_states[:, :, :, None]
        cache.update_ssm_state(self.layer_idx, new_state)
        
        # Output computation
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
        
        # Efficient reshaping
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        B = mx.tile(B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
                   (1, 1, self.heads_per_group, 1))
        C = mx.tile(C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size),
                   (1, 1, self.heads_per_group, 1))
        
        # Process sequence step by step
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
            cache: Optional[Mamba2Cache] = None,
        ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        is_incremental = cache is not None and seq_len == 1
        
        # Linear projection with efficient splitting
        projected = self.in_proj(hidden_states)
        splits = mx.split(projected, self.proj_splits, axis=-1)
        _, _, gate, conv_input, dt = splits
        
        # Apply convolution
        if is_incremental:
            conv_output = self._apply_incremental_conv(conv_input, cache)
        else:
            conv_output = self._apply_batch_conv(conv_input)
        
        # Apply activation
        conv_output = nn.silu(conv_output)
        
        # Split conv output
        hidden_states, B, C = mx.split(conv_output, self.state_splits, axis=-1)
        
        # Apply SSM
        if is_incremental:
            y = self._incremental_ssm(hidden_states, B, C, dt, cache)
        else:
            y = self._batch_ssm(hidden_states, B, C, dt)
        
        y = self.norm(y, gate)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.mixer = Mamba2Block(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(
            self,
            x: mx.array,
            cache: Optional[Mamba2Cache] = None,
        ) -> mx.array:
        if self.residual_in_fp32:
            x = x.astype(mx.float32)

        output = self.mixer(self.norm(x), cache)
        return output + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
            self, 
            x: mx.array,
            cache: Optional[Mamba2Cache] = None,
        ) -> mx.array:
        x = self.embeddings(x)
        hidden = x
        
        for layer in self.layers:
            hidden = layer(hidden, cache)
            
        return self.norm_f(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
            self,
            inputs: mx.array,
            cache: Optional[Mamba2Cache] = None
        ) -> mx.array:
        hidden = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)

        return logits

    def make_cache(self, batch_size: int = 1) -> Mamba2Cache:
        conv_dim = self.args.num_heads * self.args.head_dim + 2 * self.args.n_groups * self.args.ssm_state_size
        
        return Mamba2Cache(
            batch_size=batch_size,
            num_layers=self.args.num_hidden_layers,
            num_heads=self.args.num_heads,
            head_dim=self.args.head_dim,
            state_size=self.args.ssm_state_size,
            conv_kernel_size=self.args.conv_kernel,
            conv_dim=conv_dim
        )

    @property
    def layers(self):
        return self.backbone.layers

    def sanitize(self, weights):
        """Sanitize weights for proper loading."""
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k:
                if v.shape[-1] != 1:
                    v = v.moveaxis(2, 1)
                sanitized[k.replace("conv1d.weight", "conv_weight")] = mx.squeeze(v, axis=-1)
            elif "conv1d.bias" in k:
                sanitized[k.replace("conv1d.bias", "conv_bias")] = v
            else:
                sanitized[k] = v
        return sanitized