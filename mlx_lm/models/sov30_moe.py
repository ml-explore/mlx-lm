# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    first_k_dense_replace: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    norm_topk_prob: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    num_shared_experts: int = 0


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---- helpers ----


def _topk_lastdim(x: mx.array, k: int):
    """
    Returns (inds, vals) for top-k along last dim.
    inds: [..., k] (indices)
    vals: [..., k] (values)
    """
    inds = mx.argpartition(x, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(x, inds, axis=-1)
    # optional: sort by score desc (argpartition not sorted)
    order = mx.argsort(vals, axis=-1)[..., ::-1]
    inds = mx.take_along_axis(inds, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    return inds, vals


def _grouped_topk(scores: mx.array, top_k: int, n_group: int, topk_group: int):
    """
    Grouped top-k:
      - split experts into n_group groups
      - within each group take topk_group candidates
      - then from all candidates take global top_k
    """
    *prefix, E = scores.shape
    assert E % n_group == 0, "num_experts must be divisible by n_group"
    gsz = E // n_group

    s = scores.reshape(*prefix, n_group, gsz)  # [..., G, gsz]

    # topk within each group
    gi, gv = _topk_lastdim(s, topk_group)  # inds in [0..gsz)
    # map to global expert ids
    group_offsets = (
        mx.arange(n_group).reshape(*([1] * len(prefix)), n_group, 1) * gsz
    )
    gi_global = gi + group_offsets  # [..., G, topk_group]

    # flatten candidates
    cand_inds = gi_global.reshape(*prefix, n_group * topk_group)  # [..., C]
    cand_vals = gv.reshape(*prefix, n_group * topk_group)  # [..., C]

    # pick final top_k among candidates
    ci, cv = _topk_lastdim(cand_vals, top_k)  # ci indexes into candidates
    final_inds = mx.take_along_axis(cand_inds, ci, axis=-1)  # [..., top_k]
    final_vals = cv
    return final_inds, final_vals


class SarvamMoESparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        # Grouped routing parameters
        self.n_group = getattr(args, "n_group", 1)
        self.topk_group = getattr(args, "topk_group", 1)
        self.routed_scaling_factor = getattr(args, "routed_scaling_factor", 1.0)
        self.score_function = getattr(args, "score_function", "sigmoid")
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(torch.zeros(args.num_experts), requires_grad=True)
        else:
            self.
        # Optional shared experts
        n_shared = getattr(args, "num_shared_experts", 0) or 0
        if n_shared > 0:
            shared_inter = intermediate_size * n_shared
            self.shared_experts = MLP(dim, shared_inter)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.sigmoid(gates, axis=-1, precise=True)

        k = self.top_k

        # Use grouped top-k if n_group > 1
        if self.n_group > 1:
            inds, scores = _grouped_topk(gates, k, self.n_group, self.topk_group)
        else:
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)

        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Apply routed scaling factor
        if self.routed_scaling_factor != 1.0:
            y = y * self.routed_scaling_factor

        # Add shared experts output
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class SarvamMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

        # Determine if layer is sparse (MoE) or dense
        first_k_dense = getattr(args, "first_k_dense_replace", 0)
        mlp_only_layers = getattr(args, "mlp_only_layers", [])
        decoder_sparse_step = getattr(args, "decoder_sparse_step", 1)

        is_moe_layer = (
            layer_idx not in mlp_only_layers
            and args.num_experts > 0
            and layer_idx >= first_k_dense
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if is_moe_layer:
            self.mlp = SarvamMoESparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class SarvamMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            SarvamMoEDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SarvamMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Handle weight conversion from individual experts to switch format
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            # Shard the self attention
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            # Shard the dense MLP layers (not MoE)
            if hasattr(layer.mlp, "gate_proj"):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True
    
        return predicate
    
    @property
    def layers(self):
        return self.model.layers
