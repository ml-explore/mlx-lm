from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

BLOCK = 256

class MiniMaxText01AttentionType0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * self.head_dim * self.num_heads, bias=False)
        self.output_gate = nn.Linear(config.hidden_size, self.head_dim * self.num_heads, bias=False)
        self.norm = nn.RMSNorm(self.head_dim * self.num_heads)
        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=False)

        # for inference only
        self.offset = 0

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,  # (b, n)
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        # x: b n d
        b, n, d = x.shape
        # linear map
        qkv = nn.silu(self.qkv_proj(x))
        new_shape = qkv.size()[:-1] + (self.num_heads, -1)
        qkv = qkv.view(*new_shape)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        # for align with metaseq
        ratio = torch.exp(-slope_rate)

        # only use for the first time
        if past_key_value is None:
            slope_rate = slope_rate.to(torch.float32)
            if attn_mask is not None:
                v = v.masked_fill((1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0)
            NUM_BLOCK = (n + BLOCK - 1) // BLOCK
            b, h, n, d = q.shape
            e = v.shape[-1]
            # other
            array = torch.arange(BLOCK).to(q) + 1
            q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
            k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
            index = array[:, None] - array[None, :]
            s_index = slope_rate * index[
                None,
                None,
            ]
            s_index = torch.where(index >= 0, -s_index, float("-inf"))
            diag_decay = torch.exp(s_index)

            kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
            output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si
                qi = q[:, :, si:ei].contiguous()
                ki = k[:, :, si:ei].contiguous()
                vi = v[:, :, si:ei].contiguous()
                qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

                # diag
                qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]
                qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                block_decay = torch.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)

        else:
            kv = past_key_value
            output = []
            for i in range(n):
                kv = ratio * kv + torch.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i:i + 1],
                    v[:, :, i:i + 1],
                )
                qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i:i + 1], kv.to(q.dtype))
                output.append(qkv)
            output = torch.concat(output, dim=-2)
        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        
        # normalize
        output = self.norm(output)
        
        # gate
        output = F.sigmoid(self.output_gate(x)) * output

        # outproj
        return self.out_proj(output)
    

class MiniMaxText01AttentionType1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_dim = getattr(config, 'rotary_dim', self.head_dim)

        self.rotary_emb = ROPE(
            self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
            self,
            hidden_states,
            mask
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        ## ROPE Stuff

        ## ATTENTION STUFF with mask

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)
    

class MiniMaxText01SharedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.up_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMaxText01BlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, hidden_states):
        return self.w2(nn.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class MiniMaxText01SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MiniMaxText01BlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class MiniMaxText01DecoderLayer(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.config = config

        if attention_type == 0:
             self.self_attn = MiniMaxText01AttentionType0
        else:
             self.self_attn = MiniMaxText01AttentionType1

        self.block_sparse_moe = MiniMaxText01SparseMoeBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.postnorm = getattr(config, 'postnorm', False)
        self.layernorm_attention_alpha = getattr(config, 'layernorm_linear_attention_alpha', 1) if attention_type == 0 else getattr(config, 'layernorm_full_attention_alpha', 1)
        self.layernorm_attention_beta = getattr(config, 'layernorm_linear_attention_beta', 1) if attention_type == 0 else getattr(config, 'layernorm_full_attention_beta', 1)
        self.layernorm_mlp_alpha = getattr(config, 'layernorm_mlp_alpha', 1)
        self.layernorm_mlp_beta = getattr(config, 'layernorm_mlp_beta', 1)

        shared_intermediate = getattr(config, 'shared_intermediate_size', 0)
        self.shared_moe = False
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxText01SharedMLP(config)
            self.coefficient = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states, slope_rate):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        hidden_states = self.self_attn(hidden_states, slope_rate)

        hidden_states = residual * self.layernorm_attention_alpha  + hidden_states * self.layernorm_attention_beta

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        moe_hidden_states = self.block_sparse_moe(hidden_states)

        if self.shared_moe:
            output_mlp = self.shared_mlp(hidden_states)
            weight_fp32 = self.coefficient.weight.float()
            coef = hidden_states.to(torch.float32) @ weight_fp32.T
            coef = torch.nn.functional.sigmoid(coef).to(hidden_states.dtype)
            hidden_states = moe_hidden_states * (1 - coef) + output_mlp * coef
        else:
            hidden_states = moe_hidden_states

        hidden_states = residual * self.layernorm_mlp_alpha + moe_hidden_states * self.layernorm_mlp_beta

        return hidden_states


class MiniMaxText01Model(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn_type_list = config.attn_type_list

        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            self.layers.append(MiniMaxText01DecoderLayer(config, self.attn_type_list[i]))

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.slopes = self._build_slope_tensor(config.num_attention_heads)

    def forward(self, input_ids: torch.LongTensor = None):
        inputs_embeds = self.embed_tokens(input_ids)
        slope_rates = [self.slopes for _ in range(len(self.layers))]
        for idx, decoder_layer in enumerate(self.layers):
            slope_rate = slope_rates[idx]
            slope_rate = slope_rate * (1 - idx / (len(self.layers) - 1) + 1e-5)
            hidden_states = decoder_layer(inputs_embeds)
        return self.norm(hidden_states)


class MiniMaxText01ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxText01Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, input_ids: torch.LongTensor = None):
        return self.lm_head(self.model(input_ids))