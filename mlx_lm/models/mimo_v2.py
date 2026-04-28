# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    rope_theta: float = 5000000.0
    add_full_attention_sink_bias: bool = False
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 10000.0
    sliding_window_size: int = 128
    add_swa_attention_sink_bias: bool = True
    hybrid_layer_pattern: Optional[List[int]] = None
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: Optional[float] = None
    topk_method: str = "noaux_tc"
    partial_rotary_factor: float = 0.334
    attention_bias: bool = False
    attention_value_scale: Optional[float] = None
    layernorm_epsilon: float = 1e-5
    max_position_embeddings: int = 262144
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.hybrid_layer_pattern is None:
            self.hybrid_layer_pattern = [0] * n
        if isinstance(self.moe_layer_freq, int):
            freq = self.moe_layer_freq
            self.moe_layer_freq = [int(freq > 0 and i % freq == 0) for i in range(n)]
        elif self.moe_layer_freq is None:
            self.moe_layer_freq = [0] * n
        if len(self.hybrid_layer_pattern) != n:
            raise ValueError("hybrid_layer_pattern length must match num_hidden_layers")
        if len(self.moe_layer_freq) != n:
            raise ValueError("moe_layer_freq length must match num_hidden_layers")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sliding_window: bool):
        super().__init__()
        dim = args.hidden_size
        self.is_sliding_window = is_sliding_window
        if is_sliding_window:
            self.n_heads = args.swa_num_attention_heads
            self.n_kv_heads = args.swa_num_key_value_heads
            head_dim = args.swa_head_dim
            v_head_dim = args.swa_v_head_dim
            rope_theta = args.swa_rope_theta
            has_sinks = args.add_swa_attention_sink_bias
        else:
            self.n_heads = args.num_attention_heads
            self.n_kv_heads = args.num_key_value_heads
            head_dim = args.head_dim
            v_head_dim = args.v_head_dim
            rope_theta = args.rope_theta
            has_sinks = args.add_full_attention_sink_bias

        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * v_head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * v_head_dim, dim, bias=False)

        self.attention_sink_bias = mx.zeros((self.n_heads,)) if has_sinks else None

        self.rope = initialize_rope(
            int(args.partial_rotary_factor * head_dim),
            base=rope_theta,
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

        queries = (
            self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        )
        keys = (
            self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).swapaxes(1, 2)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.n_kv_heads, self.v_head_dim)
            .swapaxes(1, 2)
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attention_sink_bias,
        )
        return self.o_proj(output.swapaxes(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


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

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor
    return inds, scores.astype(gates.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.topk_method == "noaux_tc", "Unsupported topk method."
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor or 1.0
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))

    def __call__(self, x):
        inds, scores = group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        return inds, scores.astype(x.dtype)


class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = MoEGate(config)

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_moe: bool, is_sliding_window: bool):
        super().__init__()
        self.self_attn = Attention(config, is_sliding_window)
        self.mlp = MoE(config) if is_moe else MLP(config)
        self.is_sliding_window = is_sliding_window
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MiMoV2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        pattern = config.hybrid_layer_pattern
        moe_freq = config.moe_layer_freq
        self.layers = [
            DecoderLayer(
                config,
                is_moe=bool(moe_freq[idx]),
                is_sliding_window=bool(pattern[idx]),
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.swa_idx = pattern.index(1) if 1 in pattern else 0
        self.ga_idx = pattern.index(0) if 0 in pattern else 0
        self.sliding_window_size = config.sliding_window_size
        self.has_swa = 1 in pattern
        self.has_full = 0 in pattern

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = (
            create_attention_mask(h, cache[self.ga_idx]) if self.has_full else None
        )
        swa_mask = (
            create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window_size
            )
            if self.has_swa
            else None
        )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding_window else full_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = MiMoV2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        skip_prefixes = (
            "model.mtp.",
            "visual.",
            "audio_encoder.",
            "speech_embeddings.",
            "audio_tokenizer.",
            "model.rotary_emb.",
            "model.swa_rotary_emb.",
        )
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(skip_prefixes) and ".self_attn.rotary_emb." not in k
        }

        TP = 4
        BS = 128
        bf16 = mx.bfloat16

        def dequant_block(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=bf16)
            m, n = weight.shape
            pad_b, pad_r = (-m) % BS, (-n) % BS
            if pad_b or pad_r:
                weight = mx.pad(weight, ((0, pad_b), (0, pad_r)))
            weight = weight.reshape((m + pad_b) // BS, BS, (n + pad_r) // BS, BS)
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_b, n + pad_r
            )
            return weight[:m, :n].astype(bf16)

        def split_qkv(qkv_fp8, scale_inv, n_h, n_kv, hd, vhd):
            q_pr = (n_h // TP) * hd
            k_pr = (n_kv // TP) * hd
            v_pr = (n_kv // TP) * vhd
            actual_pr = q_pr + k_pr + v_pr
            padded_pr = -(-actual_pr // BS) * BS
            n = qkv_fp8.shape[-1]

            qkv = mx.from_fp8(qkv_fp8, dtype=bf16).reshape(TP, actual_pr, n)
            if padded_pr > actual_pr:
                qkv = mx.pad(qkv, ((0, 0), (0, padded_pr - actual_pr), (0, 0)))
            n_col_blocks = scale_inv.shape[1]
            pad_side = BS * n_col_blocks - n
            if pad_side > 0:
                qkv = mx.pad(qkv, ((0, 0), (0, 0), (0, pad_side)))

            blocked = qkv.reshape(TP * padded_pr // BS, BS, n_col_blocks, BS)
            qkv = (blocked * scale_inv[:, None, :, None]).reshape(
                TP, padded_pr, n_col_blocks * BS
            )[:, :actual_pr, :n]

            q = mx.contiguous(qkv[:, :q_pr, :]).reshape(TP * q_pr, n).astype(bf16)
            k = (
                mx.contiguous(qkv[:, q_pr : q_pr + k_pr, :])
                .reshape(TP * k_pr, n)
                .astype(bf16)
            )
            v = (
                mx.contiguous(qkv[:, q_pr + k_pr :, :])
                .reshape(TP * v_pr, n)
                .astype(bf16)
            )
            return q, k, v

        v_scale = self.args.attention_value_scale
        bake_v = v_scale is not None and v_scale != 1.0
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            qkv_key = f"{prefix}.qkv_proj.weight"
            scale_key = f"{qkv_key}_scale_inv"
            if qkv_key not in weights or scale_key not in weights:
                continue

            is_swa = bool(self.args.hybrid_layer_pattern[layer_idx])
            if is_swa:
                n_h = self.args.swa_num_attention_heads
                n_kv = self.args.swa_num_key_value_heads
                hd = self.args.swa_head_dim
                vhd = self.args.swa_v_head_dim
            else:
                n_h = self.args.num_attention_heads
                n_kv = self.args.num_key_value_heads
                hd = self.args.head_dim
                vhd = self.args.v_head_dim

            q, k, v = split_qkv(
                weights.pop(qkv_key), weights.pop(scale_key), n_h, n_kv, hd, vhd
            )
            weights[f"{prefix}.q_proj.weight"] = q
            weights[f"{prefix}.k_proj.weight"] = k
            weights[f"{prefix}.v_proj.weight"] = v

            if bake_v:
                o_key = f"{prefix}.o_proj.weight"
                w = weights.get(o_key)
                if w is not None:
                    weights[o_key] = (w * v_scale).astype(bf16)

        scale_keys = [k for k in weights if k.endswith("weight_scale_inv")]
        for sk in scale_keys:
            wk = sk[: -len("_scale_inv")]
            weights[wk] = dequant_block(weights[wk], weights.pop(sk))

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ("gate_proj", "down_proj", "up_proj"):
                expert0 = f"{prefix}.experts.0.{proj}.weight"
                if expert0 not in weights:
                    continue
                weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size))
            else:
                caches.append(KVCache())
        return caches
