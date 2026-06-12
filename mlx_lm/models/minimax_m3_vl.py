# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@partial(mx.compile, shapeless=True)
def _swiglu_oai(gate, up, alpha, limit):
    gate = mx.clip(gate, -limit, limit)
    up = mx.clip(up, -limit, limit)
    return (up + 1.0) * (gate * mx.sigmoid(gate * alpha))


class SwigluOAI(nn.Module):
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self._alpha = alpha
        self._limit = limit

    def __call__(self, x, gate):
        return _swiglu_oai(gate, x, self._alpha, self._limit)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x):
        t = x.dtype
        x = x.astype(mx.float32)
        n = mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)
        return ((x * n) * (1.0 + self.weight)).astype(t)


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 6144
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    num_hidden_layers: int = 60
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    head_dim: int = 128
    rotary_dim: int = 64
    partial_rotary_factor: float = 0.5
    vocab_size: int = 200064
    tie_word_embeddings: bool = False
    scoring_func: str = "sigmoid"
    routed_scaling_factor: float = 2.0
    use_routing_bias: bool = True
    use_qk_norm: bool = True
    use_gemma_norm: bool = True
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    moe_layer_freq: Optional[List[int]] = None
    n_shared_experts: int = 1


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict] = None
    model_type: str = "minimax_m3_vl"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextArgs.from_dict(self.text_config)
        if self.text_config is None:
            self.text_config = TextArgs()


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, alpha: float, limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self._alpha = alpha
        self._limit = limit

    def __call__(self, x):
        return self.down_proj(
            _swiglu_oai(self.gate_proj(x), self.up_proj(x), self._alpha, self._limit)
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

        activation = SwigluOAI(args.swiglu_alpha, args.swiglu_limit)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=activation,
        )

        self.shared_experts = DenseMLP(
            args.hidden_size,
            args.shared_intermediate_size,
            args.swiglu_alpha,
            args.swiglu_limit,
        )

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        ne = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        shared_out = self.shared_experts(x_flat)

        gates = self.gate(x_flat.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x_flat.dtype)

        y = self.switch_mlp(x_flat, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        y = y * self.routed_scaling_factor + shared_out

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y.reshape(*ne, -1)


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            NormClass = GemmaRMSNorm if args.use_gemma_norm else nn.RMSNorm
            self.q_norm = NormClass(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = NormClass(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)

        is_moe = bool(args.moe_layer_freq and args.moe_layer_freq[layer_idx])
        if is_moe:
            self.block_sparse_moe = SparseMoeBlock(args)
        else:
            self.mlp = DenseMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
            )
        self.is_moe = is_moe

        NormClass = GemmaRMSNorm if args.use_gemma_norm else nn.RMSNorm
        self.input_layernorm = NormClass(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = NormClass(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        if self.is_moe:
            r = r + self.block_sparse_moe(self.post_attention_layernorm(r))
        else:
            r = r + self.mlp(self.post_attention_layernorm(r))
        return r


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        NormClass = GemmaRMSNorm if args.use_gemma_norm else nn.RMSNorm
        self.norm = NormClass(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.args = config
        self.model = TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)

    def __call__(self, inputs, cache=None):
        return self.language_model(inputs, cache)

    def sanitize(self, weights):
        new = {}
        for k, v in weights.items():
            # Skip vision, MTP, and sparse attention indexer keys
            if any(
                k.startswith(p)
                for p in (
                    "visual.",
                    "vision_tower.",
                    "vision_model.",
                    "multi_modal_projector.",
                    "mm_projector.",
                )
            ):
                continue
            if any(
                f in k
                for f in (
                    ".rotary_emb.",
                    "index_q_proj",
                    "index_k_proj",
                    "index_q_norm",
                    "index_k_norm",
                    ".mtp_",
                )
            ):
                continue

            # Remap HF prefix → MLX prefix
            nk = k
            if nk.startswith("model."):
                nk = "language_model." + nk
            elif nk.startswith("lm_head."):
                nk = "language_model." + nk

            new[nk] = v
        weights = new

        # FP8 dequantize
        dequanted = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                wk = k.replace("_scale_inv", "")
                dequanted[wk] = self._dequant_fp8(weights[wk], v)
            elif k not in dequanted:
                dequanted[k] = v
        weights = dequanted

        # Expert w1/w2/w3 stacking → SwitchGLU
        args = self.args.text_config
        for l in range(args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}"
            if f"{prefix}.block_sparse_moe.experts.0.w1.weight" not in weights:
                continue
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig}.weight" in weights:
                    stacked = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig}.weight"
                        )
                        for e in range(args.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(stacked)

        return weights

    @staticmethod
    def _dequant_fp8(weight, scale_inv):
        weight = mx.from_fp8(weight, dtype=mx.bfloat16)
        bs = 128
        m, n = weight.shape
        pad_b = (-m) % bs
        pad_s = (-n) % bs
        weight = mx.pad(weight, ((0, pad_b), (0, pad_s)))
        weight = weight.reshape((m + pad_b) // bs, bs, (n + pad_s) // bs, bs)
        weight = (weight * scale_inv[:, None, :, None]).reshape(
            m + pad_b, n + pad_s
        )
        return weight[:m, :n].astype(mx.bfloat16)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.language_model.model.layers:
            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.num_attention_heads //= N
            attn.num_key_value_heads //= N

            if layer.is_moe:
                moe = layer.block_sparse_moe
                shard_inplace(
                    moe.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    moe.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    moe.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                moe.sharding_group = group

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
