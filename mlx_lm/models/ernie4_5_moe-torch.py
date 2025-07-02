from copy import deepcopy
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(attn_weights.device)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Ernie4_5_Attention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is not None else self.nums_head
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.freq_allocation = config.freq_allocation if hasattr(config, "freq_allocation") else 0
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_probs_dropout_prob", 0.0)
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_bias,
        )

        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.use_bias,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        B, L = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view(B, L, self.num_heads, -1).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(B, L, self.num_key_value_heads, -1).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(B, L, self.num_key_value_heads, -1).transpose(1, 2)

        # Cache and RoPE

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(B, L, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output


class Ernie4_5_MLP(nn.Module):
    def __init__(self, config,intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj =  nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj =  nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Ernie4_5_MoeStatics(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_experts = config.moe_num_experts
        num_experts_groups = 1

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts_groups, num_experts, dtype=torch.float32),
            requires_grad=False
        )


def topk_gate_func(
    module: nn.Module,
    hidden_states: torch.Tensor,
):
    capacity = module.get_capacity(hidden_states.shape[0])
    with torch.autocast(device_type='cuda',dtype=torch.float32):
        logits = module.gate(hidden_states.float())
    router_loss = torch.zeros([1], dtype=torch.float32, device=hidden_states.device)
    router_loss.detach()
    return logits, capacity, router_loss

class Ernie4_5_MoeMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.k = config.moe_k

        moe_intermediate_size = config.moe_intermediate_size if config.moe_intermediate_size else config.intermediate_size\
        
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False, dtype=torch.float32)

        if config.moe_gate_act == "softmax":
            self.gate_act = partial(F.softmax, dim=-1)
        elif config.moe_gate_act == "sigmoid":
            self.gate_act = F.sigmoid
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")
        
        self.experts = nn.ModuleList(
            [Ernie4_5_MLP(config,moe_intermediate_size) for i in range(config.moe_num_experts)]
        )
        
        if config.moe_use_aux_free:
            self.moe_statics = Ernie4_5_MoeStatics(config)
        
        self.use_correction_bias = config.moe_use_aux_free
        self.num_local_experts = len(self.experts)

        self.shared_experts = _init_shared_experts()

    def _init_shared_experts(self):
        cfg = deepcopy(self.config)
        if getattr(cfg, 'moe_num_shared_experts', 0) > 0:
            if getattr(cfg, 'moe_intermediate_size', None):
                cfg.intermediate_size = cfg.moe_intermediate_size * cfg.moe_num_shared_experts
            else:
                cfg.intermediate_size = cfg.intermediate_size * cfg.moe_num_shared_experts
            shared_experts = Ernie4_5_MLP(cfg, cfg.intermediate_size)
        else:
            shared_experts = None
        return shared_experts

    def forward(
        self,
        input: torch.Tensor,
    ):
        if input.dim() == 3:
            orig_shape = input.shape
            input = input.reshape(-1, input.shape[-1])
        else:
            orig_shape = None
        assert input.dim() == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"

        assert self.gate is not None

        gate_input = input

        (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            gate_prob
        ) = self.gate_and_dispatch(gate_input)

        expert_out = self.forward_experts(dispatched_input)

        combined_output = self.combine_expert_output(expert_out, combine_weights, scatter_index)

        if self.shared_experts is not None:
            shared_expert_out = self.shared_experts(gate_input)
            combined_output += shared_expert_out

        if orig_shape:
            combined_output = combined_output.reshape(orig_shape[:-1] + (combined_output.shape[-1],))

        return combined_output, combine_weights, router_loss, gate_logits

    def forward_experts(self, dispatched_input: torch.Tensor) -> torch.Tensor:
        true_experts = self.experts
        dispatched_input = dispatched_input.reshape(
            1, self.num_local_experts, -1, dispatched_input.shape[-1]
        )
        expert_outputs = []
        if isinstance(self.experts, nn.ModuleList):
            chunks = dispatched_input.permute(1, 0, 2, 3).contiguous().unbind(0)
            assert len(chunks) == len(true_experts), f"{len(chunks)}, {len(true_experts)}"
            for chunk, expert in zip(chunks, true_experts):
                expert_outputs.append(expert(chunk))
        else:
            dispatched_input = dispatched_input.permute(1, 0, 2, 3).contiguous()
            orig_shape = dispatched_input.shape
            chunks = dispatched_input.reshape(orig_shape[0], -1, orig_shape[-1])
            chunks = self.experts(chunks)
            chunks = chunks.reshape(orig_shape[:-1] + (chunks.shape[-1],)).unbind(0)
            expert_outputs.extend(chunks)
            
        expert_output = torch.stack(expert_outputs, dim=1)
        return expert_output

    def moe_gate_dispatch(
        self,
        x: torch.Tensor,                
        gate_logits: torch.Tensor,
        k: int,
        capacity: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:

        S, H = x.shape
        E = gate_logits.shape[1]
        device = x.device
        topk_prob, topk_idx = torch.topk(gate_logits, k, dim=-1)               
        combine_weights = topk_prob                                     
        expert_id = topk_idx                                            
        y = x.new_zeros((E, capacity, H))                           
        scatter_index = x.new_full((k, S), -1, dtype=torch.int32)
        
        # per-expert slot counters
        slot_counter = torch.zeros(E, dtype=torch.int32, device=device)

        for tok in range(S):
            for route in range(k):
                e = expert_id[tok, route].item()
                slot = slot_counter[e].item()
                if slot >= capacity:
                    combine_weights[tok, route] = 0.0
                    continue

                # record mapping & dispatch activation
                scatter_index[route, tok] = e * capacity + slot
                y[e, slot] = x[tok]
                slot_counter[e] += 1

        expert_offset = torch.cumsum(slot_counter, 0, dtype=torch.int64)

        return y, combine_weights, scatter_index, expert_offset, expert_id
    
    def combine_expert_output(self, expert_output: torch.Tensor, combine_weights: torch.Tensor, scatter_index: torch.Tensor) -> torch.Tensor:
        expert_output = expert_output.reshape(-1, expert_output.shape[-1])
        combined_output = self.combining(expert_output, combine_weights, scatter_index)
        return combined_output

    def combining(self, x, combine_weights, scatter_index):

        dim = x.shape[-1]

        scatter_index = scatter_index.reshape([-1])
        num_k = combine_weights.shape[-1]
        
        combine_weights = combine_weights.unsqueeze(1)

        x = x[scatter_index].reshape([-1, num_k, dim])

        return torch.matmul(combine_weights, x).squeeze(1)

    def gate_and_dispatch(self, input):
        gate_logits, capacity, router_loss = topk_gate_func(self, input)

        # capacity no use
        prob = self.gate_act(gate_logits)
        (
            dispatched_input,
            combine_weights_unnorm,
            scatter_index,
            dispatch_mask,
            _,
        ) = self.moe_gate_dispatch(input, prob,  k=self.k, capacity=capacity)
        dispatch_mask = torch.diff(F.pad(dispatch_mask, (1, 0)))

        scatter_index.detach()
        dispatch_mask.detach()

        scatter_index = scatter_index.transpose(0, 1)  # [k, s] -> [s, k]
        combine_weights = combine_weights_unnorm / torch.clamp(
            combine_weights_unnorm.sum(dim=-1, keepdim=True), min=1e-12
        )
        combine_weights = combine_weights.to(dtype=dispatched_input.dtype)

        return dispatched_input, combine_weights, dispatch_mask, scatter_index, router_loss, gate_logits, prob

    def get_capacity(self, num_tokens, cap_factor=None):
        num_experts = self.config.moe_num_experts
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.config.moe_capacity[0]
            elif num_tokens < num_experts:
                cap = self.config.moe_capacity[2]
            else:
                cap = self.config.moe_capacity[1]

        capacity = int(cap * num_tokens // num_experts)
        assert capacity > 0, f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.use_moe = config.use_moe
        self.self_attn = Ernie4_5_Attention(config, layer_idx)

        moe_layer_start_index = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        moe_layer_end_index = (
            max(config.moe_layer_end_index)
            if isinstance(config.moe_layer_end_index, (tuple, list))
            else config.moe_layer_end_index
        )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeMLP(config)
        else:
            self.mlp = Ernie4_5_MLP(config)

        self.input_layernorm = Ernie4_5_RMSNorm(config)
        self.post_attention_layernorm = Ernie4_5_RMSNorm(config)

        self.residual_add1 = Ernie4_5_ResidualWithDropout(config.hidden_dropout_prob)
        self.residual_add2 = Ernie4_5_ResidualWithDropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
        )

        hidden_states = self.residual_add1(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        return  self.residual_add2(hidden_states, residual)


class Ernie4_5_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                Ernie4_5_DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Ernie4_5_RMSNorm(config)
        self.rotary_emb = Ernie4_5_RopeEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                inputs_embeds,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states
    

class Ernie4_5_MoeForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Ernie4_5_Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,bias=config.weight_share_add_bias and config.use_bias)

    def forward(
        self,
        input_ids,
    ):
        hidden_states = self.model(
            input_ids,
        )
        return self.lm_head(hidden_states)