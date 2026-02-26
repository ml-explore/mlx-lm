"""Inference-only SarvamMoE model compatible with HuggingFace weights for SGLang."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    get_attention_tp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class SarvamMoEMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch: ForwardBatch = None, use_reduce_scatter: bool = False):
        # Guard for empty batch (DP attention idle worker)
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class SarvamMoESparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", "sigmoid")
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            reduce_results=False,  # All-reduce done after adding shared experts
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
        else:
            self.register_buffer("expert_bias", torch.zeros(config.num_experts))
        
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=True,  # Required for sigmoid scoring
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            renormalize=True,
            correction_bias=self.expert_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
            quant_config=quant_config,
            output_format=TopKOutputFormat.STANDARD,
        )
        
        if getattr(config, "num_shared_experts", None) and config.num_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            # True EP: shared experts are replicated (tp_rank=0, tp_size=1)
            if enable_moe_dense_fully_dp():
                shared_tp_rank, shared_tp_size = 0, 1
            else:
                shared_tp_rank, shared_tp_size = None, None
            self.shared_experts = SarvamMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                reduce_results=False,
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
            )
        else:
            self.shared_experts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        # Guard for empty batch (DP attention idle worker)
        if hidden_states.shape[0] == 0:
            return hidden_states
        
        # Clone input for shared experts since FusedMoE may modify hidden_states in-place
        identity = hidden_states.clone() if self.shared_experts is not None else hidden_states
        
        router_logits = F.linear(
            hidden_states.to(torch.float32),
            self.gate.weight.to(torch.float32)
        )
        
        topk_output = self.topk(hidden_states, router_logits)
        
        y = self.experts(hidden_states, topk_output)
        
        if self.shared_experts is not None:
            # Use identity (original input) for shared experts, not hidden_states which may have been modified
            shared_out = self.shared_experts(identity, forward_batch, use_reduce_scatter)
            y = y + shared_out
        
        # All-reduce after combining routed and shared experts (matches DeepseekV2 pattern)
        if self.tp_size > 1 and not use_reduce_scatter:
            y = tensor_model_parallel_all_reduce(y)
        
        return y


class SarvamMoEAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_attention_tp_size()
        
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "use_qkv_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=get_attention_tp_rank(),
            tp_size=tp_size,
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=getattr(config, "use_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=get_attention_tp_rank(),
            tp_size=tp_size,
            reduce_results=False,
        )


        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        if self.use_qk_norm:
            num_tokens = q.shape[0]
            # Reshape to (num_tokens * num_heads, head_dim) for per-head norm
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(num_tokens, self.q_size)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(num_tokens, self.kv_size)
        
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class SarvamMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = SarvamMoEAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        
        # MLP: MoE for layers >= first_k_dense_replace, else dense MLP
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        if getattr(config, "num_experts", None) and layer_id >= first_k_dense:
            self.mlp = SarvamMoESparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            # True EP: dense MLP layers are replicated (tp_rank=0, tp_size=1)
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = SarvamMoEMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,  # Match MoE pattern: all-reduce handled externally
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # DP attention support
        self.layer_id = layer_id
        self.config = config
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        
        # Determine if layer is sparse (MoE) or dense
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        self.is_layer_sparse = getattr(config, "num_experts", None) is not None and layer_id >= first_k_dense
        is_previous_layer_sparse = getattr(config, "num_experts", None) is not None and (layer_id - 1) >= first_k_dense
        
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse if layer_id > 0 else False,
        )
        
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Prepare for attention (handles layernorm + DP scatter/gather)
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        
        # Skip attention for empty batch (DP attention idle worker)
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )


        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(forward_batch)
        
        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)
        
        # All-reduce for dense MLP layers (MoE layers handle this internally)
        if not self.is_layer_sparse and self.attn_tp_size > 1 and not use_reduce_scatter:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        
        return hidden_states, residual


class SarvamMoEModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = nn.Identity()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SarvamMoEDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})
        
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class SarvamMoEForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = SarvamMoEModel(config, quant_config, add_prefix("model", prefix))
        
        # The checkpoint has a separate lm_head.weight, so don't tie
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        
        # Get EP rank and size for expert-parallel weight loading using SGLang's proper functions
        ep_size = get_moe_expert_parallel_world_size()
        ep_rank = get_moe_expert_parallel_rank()
        
        # Calculate local expert range for this EP rank
        num_experts = self.config.num_experts
        experts_per_rank = num_experts // ep_size
        local_expert_start = ep_rank * experts_per_rank
        local_expert_end = local_expert_start + experts_per_rank
        
        params_dict = dict(self.named_parameters())
        
        loaded_counts = {"qkv": 0, "expert": 0, "expert_skipped": 0, "stacked": 0, "regular": 0, "skipped": 0}
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                loaded_counts["skipped"] += 1
                continue
            
            name = name.replace(".attention.", ".self_attn.")
            name = name.replace(".dense.", ".o_proj.")
            name = name.replace(".word_embeddings.", ".embed_tokens.")
            name = name.replace(".query_layernorm.", ".q_norm.")
            name = name.replace(".key_layernorm.", ".k_norm.")
            name = name.replace(".gate.expert_bias", ".expert_bias")

            if ".query_key_value." in name:
                name = name.replace(".query_key_value.", ".qkv_proj.")
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_counts["qkv"] += 1
                continue
                
            is_expert = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                
                # Check if this expert belongs to the current EP rank
                if not (local_expert_start <= expert_id < local_expert_end):
                    loaded_counts["expert_skipped"] += 1
                    is_expert = True
                    break
                
                # Convert global expert_id to local expert_id for EP
                local_expert_id = expert_id - local_expert_start
                
                orig_name = name
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=local_expert_id)
                loaded_counts["expert"] += 1
                is_expert = True
                break
            
            if is_expert:
                continue
                

            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                orig_name = name
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_counts["stacked"] += 1
                is_stacked = True
                break
            
            if is_stacked:
                continue
                
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.warning(f"Parameter {name} not found in params_dict")
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_counts["regular"] += 1
        
        logger.info(f"Weight loading summary: QKV={loaded_counts['qkv']}, Expert={loaded_counts['expert']}, Expert_Skipped_EP={loaded_counts['expert_skipped']}, Stacked={loaded_counts['stacked']}, Regular={loaded_counts['regular']}, Skipped={loaded_counts['skipped']} (EP rank={ep_rank}/{ep_size}, local_experts={local_expert_start}-{local_expert_end-1})")


EntryClass = [SarvamMoEForCausalLM]