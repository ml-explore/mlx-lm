# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, ArraysCache, CacheList, MambaCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 2304
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 256
    routed_scaling_factor: float = 2.446
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    head_dim: int = 72
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False
    linear_attn_config: Dict = None



class KimiLinearAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.rope = nn.RoPE(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=True,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if cache is not None:
            q_pe = self.rope(q_pe, cache[-1].offset)
            k_pe = self.rope(k_pe, cache[-1].offset)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys, values = cache[-1].update_and_fetch(
                mx.concatenate([k_nope, k_pe], axis=-1), values
            )
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys = mx.concatenate([k_nope, k_pe], axis=-1)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class ShortConv(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=hidden_size,
            bias=bias
        )


    def __call__(
        self,
        x: mx.array,
        residual: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        cu_seqlens: Optional[mx.array] = None,
        **kwargs,
    ):

        B, T, D = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x * mask[..., None]


        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                cu_seqlens=cu_seqlens
            )
            return y, cache


        x_transposed = mx.transpose(x, (0, 2, 1))

        if cache is not None:
            x_padded = mx.concatenate([cache, x_transposed], axis=-1)
        else:
            padding = mx.zeros((B, D, self.padding))
            x_padded = mx.concatenate([padding, x_transposed], axis=-1)

        y = self.conv(x_padded)

        y = nn.silu(y)

        final_state = None

        final_state = x_padded[:, :, -self.kernel_size:]


        y = mx.transpose(y, (0, 2, 1))

        if residual is not None:
            y = y + residual

        return y, final_state

    def step(
        self,
        x: mx.array,
        residual: Optional[mx.array],
        cache: Optional[mx.array],
        output_final_state: bool = False,
        cu_seqlens: Optional[mx.array] = None
    ):
        """
        Single step inference (used during decoding).

        Args:
            x: Input of shape [B, 1, D] or [1, B, D] (depending on cu_seqlens)
            residual: Residual tensor
            cache: Cached states of shape [N, D, W]
            output_final_state: Whether to output final state
            cu_seqlens: Cumulative sequence lengths

        Returns:
            Tuple of (output, updated_cache)
        """
        B = x.shape[0]
        D = self.hidden_size
        W = self.kernel_size
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        if cache is None:
            cache = mx.zeros((N, D, W))

        shape = x.shape
        # Squeeze to get [B, D] or [N, D]
        x_squeezed = x.squeeze(1) if cu_seqlens is None else x.squeeze(0)

        if cache is not None:
            # Roll cache and add new input
            # cache[:, :, :-1] = cache[:, :, 1:]; cache[:, :, -1] = x_squeezed
            new_cache = mx.concatenate([cache[:, :, 1:], x_squeezed[:, :, None]], axis=-1)

            # Compute output using the conv weights
            # Shape of conv.weight: [out_channels, in_channels/groups, kernel_size]
            # For depthwise conv: [D, 1, W]
            weight = self.conv.weight.squeeze(1)  # [D, W]

            # Compute: sum(cache * weight, dim=-1)
            y = mx.sum(new_cache * weight[None, :, :], axis=-1)  # [N, D]

            if self.conv.bias is not None:
                y = y + self.conv.bias[None, :]

            y = self._apply_activation(y)
            cache[0] = new_cache
        else:
            # No cache - this shouldn't normally happen in step mode
            # Fallback: just return the input
            y = x_squeezed
            if output_final_state:
                cache = mx.zeros((N, D, W))

        # Restore original shape
        y = y.reshape(shape)

        if residual is not None:
            y = y + residual

        return y, cache



class KimiDeltaAttention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in [
            'chunk', 'fused_recurrent'], f"Not supported mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConv(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
        )
        self.k_conv1d = ShortConv(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
        )
        self.v_conv1d = ShortConv(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
        )

        # Initialize A_log parameter
        self.A_log = mx.log(mx.random.uniform(1, 16, (1, 1, self.num_heads, 1)))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = mx.zeros((projection_size,))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = nn.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        cu_seqlens: Optional[mx.array] = None,
    ):

        batch_size, q_len, _ = hidden_states.shape
        indices = None

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                indices
            )[None, ...]

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None

        if cache is not None:
            if cache[0][self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache[0][self.layer_idx]
            recurrent_state = cache[1][self.layer_idx]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            cu_seqlens=cu_seqlens
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            cu_seqlens=cu_seqlens
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            cu_seqlens=cu_seqlens
        )

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = mx.sigmoid(self.b_proj(hidden_states))

        # Reshape operations (equivalent to rearrange)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_k_dim)
        k = k.reshape(*k.shape[:-1], self.num_k_heads, self.head_k_dim)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim)


        o, recurrent_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        if cache is not None:
            cache[1][self.layer_idx] = recurrent_state
            cache[0][self.layer_idx] = (
                conv_state_q, conv_state_k, conv_state_v)

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.reshape(*g.shape[:-1], self.num_heads, self.head_dim)

        o = self.o_norm(o)
        o = o * nn.silu(g)

        o = o.reshape(o.shape[0], o.shape[1], -1)
        o = self.o_proj(o)

        if attention_mask is not None:
            o = mx.pad(o[0], [(0, 0), (0, 0), (0, 0)])

        return o


class KimiLinearMLP(nn.Module):
    def __init__(
        self, config: ModelArgs, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        down_proj = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class KimiLinearMoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = KimiLinearMLP(
                config=config, intermediate_size=intermediate_size
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = KimiLinearAttention(config)
        else:
            raise NotImplementedError

        self.mlp = (
            KimiLinearMoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else KimiLinearMLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class KimiLinearModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            KimiLinearDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        if self.pipeline_rank < extra:
            layers_per_rank += 1
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(h, cache[0])

        # Receive from the previous process in the pipeline

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            dtype = weight.dtype
            bs = 128  # block size
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        # Dequantize
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # Stack experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        # Remove multi-token prediction layer and any unused precomputed rotary freqs
        return {
            k: v
            for k, v in weights.items()
            if not k.startswith("model.layers.61") and "rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]


    def make_cache(self) -> List[Any]:
        caches = []
        for i, layer in enumerate(self.model.layers):
            conv_cache = MambaCache()
            recurrent_cache = ArraysCache(size=1)
            kv_cache = KVCache()
            caches.append(CacheList(conv_cache, recurrent_cache, kv_cache))
        return caches


