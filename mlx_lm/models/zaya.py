from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, CacheList, KVCache
from .switch_layers import SwitchGLU
from .rope_utils import initialize_rope

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "zaya"

    # ZAYA config names
    cca: bool = True
    num_query_groups: int = 2
    use_cache: bool = True
    attention_bias: bool = False
    lm_head_bias: bool = False
    vocab_size: int = 262272
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_hidden_layers: int = 80
    num_experts: int = 16
    num_attention_heads: int = 8
    head_dim: int = 128
    activation_func: str = "swiglu"
    max_position_embeddings: int = 131072
    norm_epsilon: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 106
    tie_word_embeddings: bool = True
    rope_theta: float = 5_000_000.0
    attention_dropout: float = 0.0
    moe_router_topk: int = 1
    normalization: str = "RMSNorm"
    zaya_mlp_expansion: int = 256
    zaya_use_mod: bool = True
    zaya_mod_per: int = 0
    zaya_high_prec: bool = True
    zaya_use_eda: bool = True
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    scale_residual_merge: bool = True
    fused_add_norm: bool = False
    residual_in_fp32: bool = True
    apply_rope_fusion: bool = True
    bias_activation_fusion: bool = True
    activation_func_fp8_input_store: bool = False
    sliding_window: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 0.5
    num_key_value_heads: int = 2
    clamp_temp: bool = False
    cca_time0: int = 2
    cca_time1: int = 2
    swa_layers: Optional[list[int]] = None
    swa_rotary_base: Optional[float] = None

    def __post_init__(self):
        # HF/vLLM aliases.
        if self.num_query_groups is None:
            self.num_query_groups = self.num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_query_groups
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        assert self.cca, "ZAYA prototype only supports CCA=True."
        assert self.moe_router_topk == 1, "ZAYA vLLM patch supports top-1 only."
        assert self.num_query_groups == self.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.activation_func == "swiglu"
        assert self.gated_linear_unit
        assert not self.add_bias_linear
        assert self.normalization == "RMSNorm", "Prototype supports RMSNorm only."

        # HuggingFace config currently uses `rope_scaling: false`
        # MLX initialize_rope expects either None or a dict
        if not isinstance(self.rope_scaling, dict):
            self.rope_scaling = None

        # Normalize rope parameters enough for initialize_rope users
        if self.rope_parameters is None:
            self.rope_parameters = self.rope_scaling or {"rope_type": "default"}
        if "type" in self.rope_parameters and "rope_type" not in self.rope_parameters:
            self.rope_parameters["rope_type"] = self.rope_parameters.pop("type")
        self.rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", self.partial_rotary_factor)


class ZayaArraysCache(ArraysCache):
    """Null-safe ArraysCache subclass.

    Overrides extract() to tolerate None entries, so partially initialized
    CCA state and empty MoE caches do not crash during server prompt-cache
    extraction.
    """

    def extract(self, idx):
        cache = type(self)(size=len(self.cache))
        cache.cache = [
            None if c is None else c[idx : idx + 1]
            for c in self.cache
        ]
        return cache


def _cca_empty(cache: Optional[ArraysCache]) -> bool:
    """Return True if the CCA cache has not been primed yet."""
    return cache is None or cache[0] is None or cache[1] is None


def _cca_get_conv(cache: ArraysCache) -> mx.array:
    return cache[0]


def _cca_set_conv(cache: ArraysCache, value: mx.array) -> None:
    cache[0] = value


def _cca_get_prev_hs(cache: ArraysCache) -> mx.array:
    return cache[1]


def _cca_set_prev_hs(cache: ArraysCache, value: mx.array) -> None:
    cache[1] = value


class ResidualScaling(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = mx.ones(args.hidden_size)
        self.hidden_states_bias = mx.zeros(args.hidden_size)
        if self.not_first_layer:
            self.residual_scale = mx.ones(args.hidden_size)
            self.residual_bias = mx.zeros(args.hidden_size)

    def __call__(self, residual: Optional[mx.array], hidden_states: mx.array):
        hidden_states = (
            hidden_states.astype(mx.float32) + self.hidden_states_bias.astype(mx.float32)
        ) * self.hidden_states_scale.astype(mx.float32)
        if self.not_first_layer and residual is not None:
            residual = (
                residual.astype(mx.float32) + self.residual_bias.astype(mx.float32)
            ) * self.residual_scale.astype(mx.float32)
        return residual, hidden_states


class ZayaCCA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_k_heads = args.num_query_groups
        self.num_q_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.gqa_groups = self.num_q_heads // self.num_k_heads
        self.cca_time0 = args.cca_time0
        self.cca_time1 = args.cca_time1
        self.total_padding = (args.cca_time0 - 1) + (args.cca_time1 - 1)
        self.sqrt_head_dim = self.head_dim**0.5

        self.linear_q = nn.Linear(
            self.hidden_size, self.latent_q_dim, bias=args.attention_bias
        )
        self.linear_k = nn.Linear(
            self.hidden_size, self.latent_k_dim, bias=args.attention_bias
        )
        self.val_proj1 = nn.Linear(
            self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias
        )
        self.val_proj2 = nn.Linear(
            self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias
        )

        self.conv_qk = [
            nn.Conv1d(
                in_channels=self.in_out_ch,
                out_channels=self.in_out_ch,
                kernel_size=args.cca_time0,
                groups=self.in_out_ch,
                padding=0,
                bias=True,
            ),
            nn.Conv1d(
                in_channels=self.in_out_ch,
                out_channels=self.in_out_ch,
                kernel_size=args.cca_time1,
                groups=self.num_k_heads + self.num_q_heads,
                padding=0,
                bias=True,
            ),
        ]
        self.temp = mx.zeros(self.num_k_heads)

    def _shift_hidden(self, hs: mx.array, cache: Optional[ArraysCache]):
        B, L, H = hs.shape
        if not _cca_empty(cache):
            prev = _cca_get_prev_hs(cache)[:, None, :].astype(hs.dtype)
            return mx.concatenate([prev, hs[:, :-1, :]], axis=1)
        else:
            return mx.pad(hs[:, :-1, :], [(0, 0), (1, 0), (0, 0)])

    def _conv_qk(self, qk_packed0: mx.array, cache: Optional[ArraysCache]):
        B, L, C = qk_packed0.shape
        if not _cca_empty(cache):
            cached = _cca_get_conv(cache).astype(qk_packed0.dtype)
            qk_input = mx.concatenate([cached, qk_packed0], axis=1)
        else:
            qk_input = mx.pad(qk_packed0, [(0, 0), (self.total_padding, 0), (0, 0)])

        if cache is not None:
            _cca_set_conv(cache, mx.contiguous(qk_input[:, -self.total_padding :, :]))

        qk = self.conv_qk[0](qk_input)
        qk = self.conv_qk[1](qk)
        # For a valid two-stage conv and total left padding, output length == L.
        return qk[:, -L:, :]

    def _add_grouped_qk_means(self, query, key, query_pre, key_base):
        # query:    [B, L, QH, D]
        # key:      [B, L, KH, D]
        # query_pre [B, L, QH, D]
        # key_base  [B, L, KH, D]
        B, L = query.shape[:2]
        query_pre_grouped = query_pre.reshape(
            B, L, self.num_k_heads, self.gqa_groups, self.head_dim
        ).astype(mx.float32)
        query_grouped = query.reshape(
            B, L, self.num_k_heads, self.gqa_groups, self.head_dim
        ).astype(mx.float32)
        key_base = key_base.astype(mx.float32)
        query_grouped = query_grouped + 0.5 * query_pre_grouped
        query_grouped = query_grouped + 0.5 * key_base[:, :, :, None, :]
        query = query_grouped.reshape(B, L, self.num_q_heads, self.head_dim)

        query_pre_mean = mx.mean(query_pre_grouped, axis=-2)
        key = key.astype(mx.float32) + 0.5 * query_pre_mean + 0.5 * key_base
        return query, key

    def _rms_normalize_qk(self, query, key):
        eps = 1e-12
        query = query.astype(mx.float32)
        key = key.astype(mx.float32)
        query = query * mx.rsqrt(mx.sum(query * query, axis=-1, keepdims=True) + eps)
        query = query * self.sqrt_head_dim
        key = key * mx.rsqrt(mx.sum(key * key, axis=-1, keepdims=True) + eps)
        key = key * self.sqrt_head_dim
        temp = self.temp.astype(mx.float32).reshape(1, 1, self.num_k_heads, 1)
        if self.args.clamp_temp:
            temp = mx.exp(mx.clip(temp, 1e-7, 2.0))
        key = key * temp
        return query, key

    def __call__(self, hidden_states: mx.array, cache: Optional[ArraysCache] = None):
        B, L, H = hidden_states.shape
        hs = hidden_states
        hs2 = self._shift_hidden(hs, cache)

        q = self.linear_q(hs)
        k = self.linear_k(hs)
        qk_packed0 = mx.concatenate([q, k], axis=-1)

        query_pre = q.reshape(B, L, self.num_q_heads, self.head_dim)
        key_base = k.reshape(B, L, self.num_k_heads, self.head_dim)
        qk_packed3 = self._conv_qk(qk_packed0, cache)

        if cache is not None:
            _cca_set_prev_hs(cache, mx.contiguous(hs[:, -1, :]))

        query = qk_packed3[..., : self.latent_q_dim].reshape(
            B, L, self.num_q_heads, self.head_dim
        )
        key = qk_packed3[..., self.latent_q_dim :].reshape(
            B, L, self.num_k_heads, self.head_dim
        )
        query, key = self._add_grouped_qk_means(query, key, query_pre, key_base)

        v1 = self.val_proj1(hs)
        v2 = self.val_proj2(hs2)
        value = mx.concatenate([v1, v2], axis=-1).reshape(
            B, L, self.num_k_heads, self.head_dim
        )

        query, key = self._rms_normalize_qk(query, key)
        return query.astype(hidden_states.dtype), key.astype(hidden_states.dtype), value


class ZayaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv = ZayaCCA(args)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        if isinstance(cache, CacheList):
            kv_cache, cca_cache = cache[0], cache[1]
        else:
            kv_cache, cca_cache = cache, None
        q, k, v = self.qkv(x, cache=cca_cache)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if kv_cache is not None:
            q = self.rope(q, offset=kv_cache.offset)
            k = self.rope(k, offset=kv_cache.offset)
            k, v = kv_cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class ZayaRouter(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.layer_n = layer_n
        self.hidden_size = args.hidden_size
        self.mlp_expansion = args.zaya_mlp_expansion
        self.topk = args.moe_router_topk
        self.use_mod = args.zaya_use_mod
        self.num_moe_experts = args.num_experts
        self.num_experts = args.num_experts + 1 if self.use_mod else args.num_experts
        self.router_softmax_fp32 = args.zaya_high_prec

        self.down_proj = nn.Linear(args.hidden_size, self.mlp_expansion, bias=True)

        # first router layer_n == 1 as no-EDA
        self.use_eda = bool(args.zaya_use_eda and layer_n != 1)
        self.rmsnorm_eda = nn.RMSNorm(self.mlp_expansion, eps=args.norm_epsilon)
        
        if self.use_eda:
            self.router_states_scale = mx.ones(self.mlp_expansion)

        self.router_mlp = [
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            nn.GELU(approx="precise"),
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            nn.GELU(approx="precise"),
            nn.Linear(self.mlp_expansion, self.num_experts, bias=False),
        ]

        if self.use_mod:
            # Match vLLM's initial buffer value. Loaded checkpoints may override it.
            values = [0.0] * (self.num_experts - 1) + [-1.0]
            self.balancing_biases = mx.array(values, dtype=mx.float32)
        else:
            self.balancing_biases = mx.zeros(self.num_experts, dtype=mx.float32)

    def __call__(
        self,
        hidden_states: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        hs = self.down_proj(hidden_states)
        if self.use_eda and prev_router_hidden_states is not None:
            hs = hs + prev_router_hidden_states * self.router_states_scale
        router_hidden_states_next = hs

        h = self.rmsnorm_eda(hs)
        h = self.router_mlp[0](h)
        h = self.router_mlp[1](h)
        h = self.router_mlp[2](h)
        h = self.router_mlp[3](h)
        logits = self.router_mlp[4](h)

        if self.router_softmax_fp32:
            expert_prob = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
        else:
            expert_prob = mx.softmax(logits, axis=-1, precise=True)

        biased = mx.stop_gradient(expert_prob.astype(mx.float32)) + self.balancing_biases
        # Top-1 only
        expert_choice = mx.argmax(biased, axis=-1)[..., None]
        route_prob = mx.take_along_axis(expert_prob, expert_choice, axis=-1).astype(
            hidden_states.dtype
        )
        return route_prob, expert_choice, router_hidden_states_next


class ZayaBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.layer_n = layer_n
        self.router = ZayaRouter(args, layer_n)
        self.experts = SwitchGLU(
            args.hidden_size,
            args.ffn_hidden_size // 2,
            args.num_experts,
            bias=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        probs, indices, router_hidden_states = self.router(
            hidden_states, prev_router_hidden_states
        )
        if self.args.zaya_use_mod:
            clamped = mx.minimum(indices, self.args.num_experts - 1)
            expert_y = (self.experts(hidden_states, clamped) * probs[..., None]).squeeze(
                -2
            )
            mod_y = hidden_states * probs.squeeze(-1)[..., None]
            mod_mask = indices.squeeze(-1)[..., None] == self.args.num_experts
            y = mx.where(mod_mask, mod_y, expert_y)
        else:
            y = (self.experts(hidden_states, indices) * probs[..., None]).squeeze(-2)
        return y, router_hidden_states


class ZayaDecoderATTLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.layer_type = "attention"
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        self.self_attn = ZayaAttention(args)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)
        self.args = args

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        
        # accumulate residual per-layer, then norm the sum.
        if residual is not None:
            residual = residual.astype(mx.float32) + hidden_states.astype(mx.float32)
        else:
            residual = hidden_states.astype(mx.float32)
        x = self.input_norm(residual.astype(hidden_states.dtype))
        hidden_states = self.self_attn(x, mask=mask, cache=cache)
        return hidden_states, residual, prev_router_hidden_states


class ZayaDecoderMLPLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.layer_type = "moe"
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        self.zaya_block = ZayaBlock(args, layer_n)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)
        self.args = args

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        
        # accumulate residual per-layer, then norm the sum.
        if residual is not None:
            residual = residual.astype(mx.float32) + hidden_states.astype(mx.float32)
        else:
            residual = hidden_states.astype(mx.float32)
        x = self.input_norm(residual.astype(hidden_states.dtype))
        hidden_states, prev_router_hidden_states = self.zaya_block(
            x, prev_router_hidden_states
        )
        return hidden_states, residual, prev_router_hidden_states


class ZayaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            ZayaDecoderMLPLayer(args, i) if i % 2 == 1 else ZayaDecoderATTLayer(args, i)
            for i in range(args.num_hidden_layers)
        ]
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, args.num_hidden_layers)
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        hidden_states = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        # find the first attention layer's KV cache for the attention mask.
        attn_cache = None
        for c in cache:
            if c is not None:
                if isinstance(c, CacheList):
                    attn_cache = c[0]
                else:
                    attn_cache = c
                break
        mask = create_attention_mask(hidden_states, attn_cache)

        residual = None
        prev_router_hidden_states = None
        for layer, c in zip(self.layers, cache):
            layer_mask = mask if layer.layer_type == "attention" else None
            hidden_states, residual, prev_router_hidden_states = layer(
                hidden_states,
                residual,
                mask=layer_mask,
                cache=c,
                prev_router_hidden_states=prev_router_hidden_states,
            )

        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        
        # final merge of residual + hidden_states before final norm
        if residual is not None:
            hidden_states = hidden_states.astype(mx.float32) + residual.astype(mx.float32)
        else:
            hidden_states = hidden_states.astype(mx.float32)

        # Use the norm weight dtype, not the embedding dtype (which may be quantized)
        return self.final_norm(hidden_states.astype(self.final_norm.weight.dtype))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ZayaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=args.lm_head_bias
            )

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.args.zaya_high_prec:
            out = out.astype(mx.float32)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            CacheList(KVCache(), ZayaArraysCache(size=2))
            if l.layer_type == "attention"
            else ZayaArraysCache(size=0)
            for l in self.layers
        ]

    # this is necessary for convert/quant to work correctly with vLLM/Megatron checkpoints provided
    def sanitize(self, weights):
        """Normalize checkpoint paths and convert local_experts to SwitchGLU format."""
        # Path aliasing for vLLM/Megatron variants.
        aliased = {}
        for k, v in weights.items():
            nk = k.replace(".self_attn.cca.", ".self_attn.qkv.")
            nk = re.sub(
                r"(model\.layers\.\d+)\.moe\.zaya_block\.",
                r"\1.zaya_block.",
                nk,
            )
            nk = re.sub(
                r"(model\.layers\.\d+)\.moe\.experts\.",
                r"\1.zaya_block.experts.",
                nk,
            )
            nk = re.sub(
                r"(model\.layers\.\d+)\.moe\.router\.",
                r"\1.zaya_block.router.",
                nk,
            )
            nk = re.sub(
                r"(model\.layers\.\d+)\.experts\.",
                r"\1.zaya_block.experts.",
                nk,
            )
            nk = re.sub(
                r"(model\.layers\.\d+)\.router\.",
                r"\1.zaya_block.router.",
                nk,
            )
            nk = re.sub(
                r"(model\.layers\.\d+)\.moe\.(input_norm|res_scale\.)",
                r"\1.\2",
                nk,
            )
            aliased[nk] = v
        weights = aliased

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights.pop("lm_head.bias", None)

        # Conv1d PyTorch [out, in/groups, kernel] -> MLX [out, kernel, in/groups].
        for k, v in list(weights.items()):
            if ".conv_qk." in k and k.endswith(".weight") and v.ndim == 3:
                if v.shape[-1] in (self.args.cca_time0, self.args.cca_time1):
                    weights[k] = v.moveaxis(2, 1)

        # convert local expert format to SwitchGLU stacked format
        local_fc1 = [k for k in weights if ".local_experts." in k and k.endswith(".linear_fc1.weight")]
        if local_fc1:
            any_converted = False
            for layer_n in range(self.args.num_hidden_layers):
                if layer_n % 2 == 0:
                    continue
                base = f"model.layers.{layer_n}.zaya_block.experts"
                gate, up, down = [], [], []
                present = True
                for e in range(self.args.num_experts):
                    fc1_key = f"{base}.local_experts.{e}.linear_fc1.weight"
                    fc2_key = f"{base}.local_experts.{e}.linear_fc2.weight"
                    if fc1_key not in weights or fc2_key not in weights:
                        present = False
                        break
                    fc1 = weights.pop(fc1_key)
                    half = fc1.shape[0] // 2
                    gate.append(fc1[:half])
                    up.append(fc1[half:])
                    down.append(weights.pop(fc2_key))
                if present:
                    weights[f"{base}.gate_proj.weight"] = mx.stack(gate)
                    weights[f"{base}.up_proj.weight"] = mx.stack(up)
                    weights[f"{base}.down_proj.weight"] = mx.stack(down)
                    any_converted = True
            if not any_converted:
                raise ValueError(
                    "Checkpoint contains local_experts weights but they could not be "
                    "packed into SwitchGLU format. This model layout is not supported."
                )

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # keep router numerics, norms, scales and biases unquantized.
            if "router" in path:
                return False
            if ".qkv.temp" in path or "balancing_biases" in path:
                return False
            # norm parameters should never be quantized.
            if "input_norm" in path or "final_norm" in path or "rmsnorm" in path:
                return False
            # residual scaling parameters should stay in fp32.
            if "res_scale" in path:
                return False
            return True

        return predicate
