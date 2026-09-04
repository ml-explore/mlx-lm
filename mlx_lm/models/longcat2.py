# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, CacheList, KVCache
from .longcat_flash import LongcatFlashMLP, LongcatFlashMoE
from .longcat_flash_ngram import NgramEmbedding
from .mla import MultiLinear
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    ffn_hidden_size: int
    moe_topk: int
    expert_ffn_hidden_size: int
    n_routed_experts: int
    zero_expert_num: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    model_type: str = "longcat2"
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    index_k_norm_type: str = "rms"
    index_topk_pattern: Optional[list] = None
    index_init_tokens: int = 16
    index_local_tokens: int = 1024
    cli_factor: int = 2
    eos_token_id: Optional[int] = 2
    indexer_rope_interleave: bool = True
    rope_interleave: Optional[bool] = None
    use_longcat_dsa: bool = True
    oe_neighbor_num: Optional[int] = 5
    oe_split_num: Optional[int] = 4
    oe_vocab_size_ratio: Optional[float] = 100.567
    ngram_vocab_size_ratio: Optional[float] = None
    emb_neighbor_num: int = 5
    emb_split_num: int = 4
    attention_bias: bool = False
    zero_expert_type: str = "identity"
    norm_topk_prob: bool = False
    router_bias: bool = False
    attention_method: str = "MLA"
    rope_scaling: Optional[Dict] = None

    def __post_init__(self):
        if self.rope_interleave is not None:
            self.indexer_rope_interleave = self.rope_interleave
        use_oe = (
            self.oe_vocab_size_ratio is not None
            and self.oe_neighbor_num is not None
            and self.oe_split_num is not None
        )
        if self.ngram_vocab_size_ratio is None and use_oe:
            self.ngram_vocab_size_ratio = self.oe_vocab_size_ratio
            self.emb_neighbor_num = self.oe_neighbor_num
            self.emb_split_num = self.oe_split_num
        if self.index_topk_pattern is None and self.use_longcat_dsa:
            self.index_topk_pattern = [
                "C" if i % self.cli_factor == 0 else "S"
                for i in range(self.num_layers * 2)
            ]


def _indexer_is_shared(args: ModelArgs, global_idx: int) -> bool:
    if not args.use_longcat_dsa:
        return True
    if args.index_topk_pattern is not None:
        return args.index_topk_pattern[global_idx] == "S"
    return False


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.index_init_tokens = args.index_init_tokens
        self.index_local_tokens = args.index_local_tokens
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        if args.index_k_norm_type == "rms":
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        else:
            self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.indexer_rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        b, s, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.wk(x)
        k = self.k_norm(k)
        k = mx.reshape(k, (b, 1, s, self.head_dim))

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
        if k.shape[2] <= self.index_topk:
            return None

        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
        if scores.size <= 2**31:
            scores = weights[..., None, :] @ scores.transpose(0, 2, 1, 3)
            scores = scores.transpose(0, 2, 1, 3)
        else:
            scores = scores * weights.swapaxes(-1, -2)[..., None]
            summed = scores[:, 0:1]
            for h in range(1, scores.shape[1]):
                summed = summed + scores[:, h : h + 1]
            scores = summed
        if self.index_init_tokens or self.index_local_tokens:
            S = scores.shape[-1]
            kv_pos = mx.arange(S)
            lengths = mx.arange(S - s + 1, S + 1)
            force = (kv_pos[None] < self.index_init_tokens) | (
                kv_pos[None] >= lengths[:, None] - self.index_local_tokens
            )
            scores = mx.where(force, mx.array(mx.inf, scores.dtype), scores)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]


class Longcat2Attention(nn.Module):
    def __init__(self, args: ModelArgs, global_idx: int):
        super().__init__()
        self.skip_topk = _indexer_is_shared(args, global_idx)
        self.num_attention_heads = args.num_attention_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.v_head_dim = args.v_head_dim

        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            args.hidden_size, self.q_lora_rank, bias=args.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_attention_heads * self.qk_head_dim,
            bias=False,
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_attention_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_attention_heads
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.mla_scale_q_lora = (
            (args.hidden_size / self.q_lora_rank) ** 0.5
            if args.mla_scale_q_lora
            else None
        )
        self.mla_scale_kv_lora = (
            (args.hidden_size / self.kv_lora_rank) ** 0.5
            if args.mla_scale_kv_lora
            else None
        )

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = args.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        if not self.skip_topk:
            self.indexer = Indexer(args)

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        topk_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        if self.mla_scale_q_lora is not None:
            qr = qr * self.mla_scale_q_lora
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_attention_heads, self.qk_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        if self.mla_scale_kv_lora is not None:
            kv_latent = kv_latent * self.mla_scale_kv_lora

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None, None]

        if not self.skip_topk:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        if not self.skip_topk and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_indices


class Longcat2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mlp = LongcatFlashMoE(args)

        self.self_attn = [Longcat2Attention(args, layer_idx * 2 + i) for i in range(2)]
        self.mlps = [LongcatFlashMLP(args, False) for _ in range(2)]
        self.input_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = x
        shortcut_mlp_output = None

        if cache is None:
            cache = (None, None)

        topk_indices = None
        for i in range(2):
            residual = hidden_states

            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states, topk_indices = self.self_attn[i](
                hidden_states, mask=mask, cache=cache[i], topk_indices=topk_indices
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)

            if i == 0:
                shortcut_mlp_output = self.mlp(hidden_states)

            hidden_states = self.mlps[i](hidden_states)
            hidden_states = residual + hidden_states

            if i == 1:
                hidden_states = hidden_states + shortcut_mlp_output

        return hidden_states


class Longcat2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.num_layers
        self.ngram_embeddings = NgramEmbedding(args)
        self.layers = [
            Longcat2DecoderLayer(args, idx) for idx in range(args.num_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * (self.num_layers + 1)

        h = self.ngram_embeddings(input_ids, cache=cache[0])

        first = cache[1]
        main_kv = first[0][0] if first is not None else None
        mask = create_attention_mask(h, main_kv, return_array=True)

        for layer, c in zip(self.layers, cache[1:]):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Longcat2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("classifier"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def sanitize(self, weights):
        def _keep(k):
            if "mtp" in k:
                return False
            parts = k.split(".")
            if (
                len(parts) >= 3
                and parts[1] == "layers"
                and parts[2].isdigit()
                and int(parts[2]) >= self.args.num_layers
            ):
                return False
            return True

        weights = {k: v for k, v in weights.items() if _keep(k)}

        def dequant(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
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
            return weight[:m, :n].astype(mx.bfloat16)

        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                wk = k.replace("_scale_inv", "")
                new_weights[wk] = dequant(weights[wk], v)
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        renames = {}
        for k in list(weights.keys()):
            if k == "model.embed_tokens.weight":
                renames[k] = "model.ngram_embeddings.word_embeddings.weight"
            elif k.startswith("model.oe_embed_tokens"):
                idx = k[len("model.oe_embed_tokens") :]
                idx = idx.replace(".weight", "").lstrip(".")
                renames[k] = f"model.ngram_embeddings.embedders.{idx}.weight"
            elif k.startswith("model.oe_embed_proj"):
                idx = k[len("model.oe_embed_proj") :]
                idx = idx.replace(".weight", "").lstrip(".")
                renames[k] = f"model.ngram_embeddings.post_projs.{idx}.weight"
        for old, new in renames.items():
            weights[new] = weights.pop(old)

        for l in range(self.args.num_layers):
            prefix = f"model.layers.{l}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        for l in range(self.args.num_layers):
            for i in range(2):
                prefix = f"model.layers.{l}.self_attn.{i}"
                kv_b_key = f"{prefix}.kv_b_proj.weight"
                if kv_b_key in weights:
                    num_heads = self.args.num_attention_heads
                    head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                    quantized = f"{prefix}.kv_b_proj.scales" in weights
                    v = weights.pop(kv_b_key)

                    if quantized:
                        dims = self.args.kv_lora_rank
                        scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                        biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                        bits = (v.shape[-1] * 32) // dims
                        group_size = dims // scales.shape[-1]
                        v = mx.dequantize(
                            v, scales, biases, bits=bits, group_size=group_size
                        )

                    v = v.reshape(num_heads, head_dim, -1)
                    wk = mx.contiguous(
                        v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                    )
                    wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])

                    if quantized:
                        wk, wk_s, wk_b = mx.quantize(
                            wk, bits=bits, group_size=group_size
                        )
                        wv, wv_s, wv_b = mx.quantize(
                            wv, bits=bits, group_size=group_size
                        )
                        weights[f"{prefix}.embed_q.scales"] = wk_s
                        weights[f"{prefix}.embed_q.biases"] = wk_b
                        weights[f"{prefix}.unembed_out.scales"] = wv_s
                        weights[f"{prefix}.unembed_out.biases"] = wv_b

                    weights[f"{prefix}.embed_q.weight"] = wk
                    weights[f"{prefix}.unembed_out.weight"] = wv

        return weights

    def make_cache(self):
        caches = [ArraysCache(size=1)]
        for layer in self.layers:
            sub_caches = []
            for attn in layer.self_attn:
                if attn.skip_topk:
                    sub_caches.append(CacheList(KVCache()))
                else:
                    sub_caches.append(CacheList(KVCache(), KVCache()))
            caches.append(CacheList(*sub_caches))
        return caches

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        for layer in self.model.layers:
            for attn in layer.self_attn:
                attn.q_b_proj = shard_linear(
                    attn.q_b_proj, "all-to-sharded", group=group
                )
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
                attn.num_attention_heads //= N
                num_heads = attn.num_attention_heads
                sh = rank * num_heads
                eh = sh + num_heads

                def shard_heads(w):
                    return w[sh:eh]

                attn.embed_q.apply(shard_heads)
                attn.unembed_out.apply(shard_heads)

            for mlp in layer.mlps:
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

            layer.mlp.sharding_group = group
            shard_inplace(layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
