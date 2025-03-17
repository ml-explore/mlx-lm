from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Mamba2Config():
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


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class Mamba2Cache:
    def __init__(
        self, config: Mamba2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.intermediate_size + 2 * self.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)
    

class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Mamba2RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    def forward(self, input_states, cache_params: Optional[Mamba2Cache]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size-self.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )

        # 2. Convolution sequence transformation
        cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)

        # We need to guarantee that anything regarding the cache is on the same device
        conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

        hidden_states_B_C = torch.sum(
            conv_states * self.conv1d.weight.squeeze(1), dim=-1
        )
        if self.use_conv_bias:
            hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
        hidden_states_B_C = nn.SiLU(hidden_states_B_C)

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log) # [num_heads]

        # We need to guarantee that anything regarding the cache is on the same device
        cache_device = cache_params.ssm_states.device

        # Note: there is no need to pad parameter matrices here, as there is just one new token
        # for batched generation
        dt = dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
        # [num_heads] -> [num_heads, head_dim]
        dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

        dt = torch.nn.functional.softplus(dt + dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
        A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
        # [bsz, num_heads, head_dim, state_size]
        dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

        # Discretize B
        # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
        # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
        B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])
        # [bsz, num_heads, head_dim, state_size]
        dB = dt[..., None] * B[..., None, :]

        # Discretize x into dB
        # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
        hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
        dBx = (dB * hidden_states[..., None]).to(device=cache_device)

        # State calculation
        cache_params.update_ssm_state(
            layer_idx=self.layer_idx,
            new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx
        )

        # Subsequent output
        # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
        C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])
        # [bsz, num_heads, head_dim]

        ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
        # Reshape ssm_states to merge the first two dimensions
        ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
        C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
        y = torch.bmm(ssm_states_reshaped, C_reshaped)
        y = y.view(batch_size, self.num_heads, self.head_dim)

        # D skip connection
        # [num_heads] -> [num_heads, head_dim]
        D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
        y = (y + hidden_states * D).to(y.dtype)

        # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
        y = y.reshape(batch_size, -1)[:, None, ...]

        scan_output = self.norm(y, gate)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states


class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Mamba2Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        return hidden_states



class Mamba2Model(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.embeddings(input_ids)

        if cache_params is None:
            cache_params = Mamba2Cache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        for mixer_block in self.layers:
            hidden_states = mixer_block(
                inputs_embeds,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )

        return self.norm_f(hidden_states)


class Mamba2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        mamba2_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba2_outputs[0]
        return self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()