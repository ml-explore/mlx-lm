from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn


# Copied from transformers.models.jetmoe.modeling_jetmoe.JetMoeParallelExperts with JetMoe->GraniteMoe
class GraniteMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Initialize the GraniteMoeParallelExperts module.
        The experts weights are stored in [num_experts, output_size, input_size] format. Such that it's compatible with
        many MoE libraries, such as [Megablock](https://github.com/databricks/megablocks) and
        [ScatterMoE](https://github.com/shawntan/scattermoe), as well as the
        [MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
        used in vllm.

        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs, expert_size):
        """
        Forward pass of the GraniteMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        results = torch.cat(output_list, dim=0)
        return results


# Copied from transformers.models.jetmoe.modeling_jetmoe.JetMoeTopKGating with JetMoe->GraniteMoe
class GraniteMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        """
        Initialize the top-k gating mechanism.
        Args:
            input_size (`int`):
                Size of the input.
            num_experts (`int`):
                Number of experts.
            top_k (`int`):
                Number of top experts to select.
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, hidden_states):
        # compute the top_k routing decision
        logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

        # compute number of input given to each expert
        zeros = torch.zeros(
            [top_k_gates.size(0), self.num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
        )  # [num_tokens, num_experts]
        gates = zeros.scatter(1, top_k_indices, 1)  # [num_tokens, num_experts]
        expert_size = gates.long().sum(0)  # [num_experts,]
        # (This cause torch.compile to fail with `torch._dynamo.exc.Unsupported: Backend compiler failed with a fake tensor exception at`)
        # (and `DataDependentOutputException`)
        expert_size = expert_size.tolist()

        # sort and group input tokens according to expert assignment
        top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
        batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

        return index_sorted_experts, batch_index, batch_gates, expert_size


class GraniteMoeMoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: GraniteMoeConfig):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.intermediate_size
        self.input_linear = GraniteMoeParallelExperts(config.num_local_experts, self.input_size, self.hidden_size * 2)
        self.output_linear = GraniteMoeParallelExperts(config.num_local_experts, self.hidden_size, self.input_size)

        self.router = GraniteMoeTopKGating(
            input_size=self.input_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(self, layer_input):
        """
        Forward pass of the mixture of experts layer.

        Args:
            layer_input (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Output tensor.
            Tensor:
                Router logits.
        """
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        _, batch_index, batch_gates, expert_size = self.router(layer_input)

        expert_inputs = layer_input[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = nn.functional.silu(chunked_hidden_states[0]) * chunked_hidden_states[1]
        expert_outputs = self.output_linear(hidden_states, expert_size)

        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        return layer_output


# copied from transformers.models.granite.modeling_granite.GraniteAttention with Granite->GraniteMoe
class GraniteMoeAttention(nn.Module):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.scaling = config.attention_multiplier

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE and Cache

        attn_output, attn_weights = nn.functional.scaled_dot_product_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GraniteMoeDecoderLayer(nn.Module):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GraniteMoeAttention(config=config)

        if config.num_local_experts > 0:
            self.block_sparse_moe = GraniteMoeMoE(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states * self.residual_multiplier

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)

        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GraniteMoeModel(nn.Module):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__(config)
        self.embedding_multiplier = config.embedding_multiplier
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GraniteMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
    ):
        hidden_states = self.embed_tokens(input_ids) * self.embedding_multiplier

        # create mask and cache objects

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__(config)
        self.config = config

        self.model = GraniteMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # do _tied_weights_keys

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        return self.lm_head(hidden_states) / self.config.logits_scaling