# Copyright © 2026 Apple Inc.

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_unflatten

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .deepseek_v3 import DeepseekV3MLP
from .mistral4 import mistral4_expert_select
from .mla import MultiLinear
from .pipeline import PipelineMixin
from .rope_utils import YarnRoPE
from .switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear


@dataclass
class TextArgs(BaseModelArgs):
    vocab_size: int = 204800
    hidden_size: int = 6144
    intermediate_size: int = 24576
    moe_intermediate_size: int = 6144
    num_hidden_layers: int = 72
    num_attention_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    rms_norm_eps: float = 1e-5
    rope_parameters: Optional[Dict] = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    first_k_dense_replace: int = 1
    mlp_layer_types: Optional[List[str]] = None
    indexer_types: Optional[List[str]] = None
    no_rope_layers: Optional[List[int]] = None
    no_rope_layer_interval: int = 4
    kv_source_layers: Optional[List[int]] = None
    final_logit_softcapping: Optional[float] = 30.0

    def __post_init__(self):
        self.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2048.0,
            "original_max_position_embeddings": 512,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
            **(self.rope_parameters or {}),
        }
        n = self.num_hidden_layers
        if self.no_rope_layers is None:
            interval = self.no_rope_layer_interval
            self.no_rope_layers = [int((i + 1) % interval != 0) for i in range(n)]
        if self.kv_source_layers is None:
            self.kv_source_layers = list(range(n))
        if self.indexer_types is None:
            self.indexer_types = ["full"] * n
        if self.mlp_layer_types is None:
            dense = min(self.first_k_dense_replace, n)
            self.mlp_layer_types = ["dense"] * dense + ["sparse"] * (n - dense)
        for i, src in enumerate(self.kv_source_layers):
            if src > i or self.kv_source_layers[src] != src:
                raise ValueError(f"Layer {i} cannot share the KV cache of layer {src}.")
        if self.indexer_types[0] != "full":
            raise ValueError("The first layer must run its own indexer.")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict = field(default_factory=dict)
    quantization_config: Optional[dict] = None


def _log_positions(offset, length):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]
    # Left padding of a batch has negative positions.
    positions = mx.maximum(mx.arange(length) + offset, 0)
    return mx.log(positions.astype(mx.float32) + 1)


def _fp8_as_mxfp8(weight, scale, block=128):
    # Power-of-two block scales are exact E8M0 scales, one per row and 32 columns.
    (m, n), (sm, sn) = weight.shape, scale.shape
    if (sm, sn) != (-(-m // block), -(-n // block)) or n % 32:
        raise ValueError(f"FP8 blocks of {weight.shape} / {scale.shape} are not mxfp8.")
    e8m0 = (mx.log2(scale.astype(mx.float32)) + 127).astype(mx.uint8)
    # The last block in each dimension can be smaller than the block size.
    e8m0 = mx.repeat(e8m0, block, axis=0)[:m]
    e8m0 = mx.repeat(e8m0, block // 32, axis=1)[:, : n // 32]
    return weight.view(mx.uint32), e8m0


def _mxfp8_dequantize(weight, scales):
    return mx.dequantize(
        weight, scales, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16
    )


class Fp8Linear(nn.QuantizedLinear):
    """FP8 checkpoint weights, run as mxfp8.

    Unlike other quantized layers, ``convert -q`` quantizes it again, since a
    bf16 copy of this model does not fit in memory or on disk.
    """

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__(input_dims, output_dims, False, 32, 8, "mxfp8")

    def to_quantized(self, group_size=None, bits=None, mode="affine", **kwargs):
        out_dims, packed = self.weight.shape
        linear = nn.Linear(packed * 4, out_dims, bias=False)
        linear.weight = _mxfp8_dequantize(self.weight, self.scales)
        return linear.to_quantized(
            group_size=group_size, bits=bits, mode=mode, **kwargs
        )


class Fp8SwitchLinear(QuantizedSwitchLinear):
    """FP8 checkpoint experts, run as mxfp8. See Fp8Linear."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__(input_dims, output_dims, num_experts, False, 32, 8, "mxfp8")

    def to_quantized(self, group_size=64, bits=4, mode="affine", **kwargs):
        num_experts, out_dims, packed = self.weight.shape
        linear = SwitchLinear(packed * 4, out_dims, num_experts, bias=False)
        linear.weight = _mxfp8_dequantize(self.weight, self.scales)
        return linear.to_quantized(
            group_size=group_size, bits=bits, mode=mode, **kwargs
        )


class LeRoPE(YarnRoPE):
    """Interleaved YaRN RoPE with a learned scale for each frequency."""

    def __init__(self, dims: int, rope_parameters: dict):
        if rope_parameters["rope_type"] != "yarn":
            raise ValueError("Only yarn RoPE is supported.")
        if rope_parameters["attention_factor"] != 1.0:
            raise ValueError("Only yarn without magnitude scaling is supported.")
        super().__init__(
            dims,
            traditional=True,
            base=rope_parameters["rope_theta"],
            scaling_factor=rope_parameters["factor"],
            original_max_position_embeddings=rope_parameters[
                "original_max_position_embeddings"
            ],
            beta_fast=rope_parameters["beta_fast"],
            beta_slow=rope_parameters["beta_slow"],
            mscale=0,
        )
        self.alpha = mx.zeros((dims // 2,))

    def freqs(self):
        # exp(alpha) scales the inverse frequencies, so it divides the wavelengths.
        return self._freqs * mx.exp(-self.alpha.astype(mx.float32))


class SoftmaxScaling(nn.Module):
    """Per-head query scale s * log(position + 1) + b."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.scale = mx.zeros((num_heads,))
        self.bias = mx.ones((num_heads,))

    def __call__(self, q, log_pos):
        # q is [B, H, L, D] and log_pos is [L] or [B, L].
        scale = self.scale[:, None] * log_pos[..., None, :] + self.bias[:, None]
        return q * scale[..., None].astype(q.dtype)


class Indexer(nn.Module):
    def __init__(self, args: TextArgs, use_rope: bool):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.use_rope = use_rope
        # No q-LoRA, so the queries come from the hidden states.
        self.q_input_norm = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.wq_b = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.softmax_scaling = SoftmaxScaling(self.n_heads)

    def __call__(self, x, mask, rope, log_pos, cache=None, block=4096):
        B, L, _ = x.shape
        q = self.wq_b(self.q_input_norm(x))
        q = q.reshape(B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        q = self.softmax_scaling(q, log_pos)
        k = self.k_norm(self.wk(x)).reshape(B, 1, L, self.head_dim)
        if self.use_rope:
            # The rope dims come first in the indexer heads.
            q = rope(q)
            k = rope(k)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), k.dtype))
        N = k.shape[2]
        if N <= self.index_topk:
            return None

        w = self.weights_proj(x.astype(mx.float32))
        w = (w * (self.n_heads**-0.5 * self.head_dim**-0.5))[:, :, None, :]
        q = q.swapaxes(1, 2).astype(mx.float32)  # [B, L, H, D]
        k = k.astype(mx.float32)

        best_v = best_i = None
        for s0 in range(0, N, block):
            s1 = min(N, s0 + block)
            sc = q @ k[:, :, s0:s1].swapaxes(-1, -2)  # [B, L, H, blk]
            sc = (w @ mx.maximum(sc, 0)).swapaxes(1, 2)
            if mask is not None:
                sc = mx.where(mask[..., s0:s1], sc, -mx.inf)
            ids = mx.broadcast_to(mx.arange(s0, s1), sc.shape)
            if best_v is not None:
                sc = mx.concatenate([best_v, sc], axis=-1)
                ids = mx.concatenate([best_i, ids], axis=-1)
            if sc.shape[-1] > self.index_topk:
                pos = mx.argpartition(sc, kth=-self.index_topk, axis=-1)
                pos = pos[..., -self.index_topk :]
                sc = mx.take_along_axis(sc, pos, axis=-1)
                ids = mx.take_along_axis(ids, pos, axis=-1)
            best_v, best_i = sc, ids
        return mx.stop_gradient(best_i)  # [B, 1, L, topk]


class Attention(nn.Module):
    """Absorbed MLA with DSA top-k selection, NoPE layers and KV sharing."""

    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_attention_heads
        self.use_rope = bool(args.no_rope_layers[layer_idx])
        self.kv_source = args.kv_source_layers[layer_idx]
        self.owns_kv = self.kv_source == layer_idx
        self.kv_lora_rank = args.kv_lora_rank
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        # NoPE layers use the rope dims as nope dims.
        self.rope_dim = args.qk_rope_head_dim if self.use_rope else 0
        self.nope_dim = self.q_head_dim - self.rope_dim
        self.scale = self.q_head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.q_head_dim, bias=False
        )
        if self.owns_kv:
            self.kv_a_proj_with_mqa = nn.Linear(
                args.hidden_size, self.kv_lora_rank + self.rope_dim, bias=False
            )
            self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        # kv_b_proj is absorbed, so the cache holds the compressed latent.
        self.embed_q = MultiLinear(self.nope_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, args.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * args.v_head_dim, args.hidden_size, bias=False
        )
        self.softmax_scaling = SoftmaxScaling(self.num_heads)
        if args.indexer_types[layer_idx] == "full":
            self.indexer = Indexer(args, self.use_rope)
        else:
            self.indexer = None

    def __call__(self, x, mask, rope, log_pos, cache, shared_kv, prev_topk):
        B, L, _ = x.shape
        has_indexer = self.indexer is not None
        mla_cache = cache[0] if cache is not None and self.owns_kv else None
        index_cache = None
        if cache is not None and has_indexer:
            index_cache = cache[1 if self.owns_kv else 0]

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = self.softmax_scaling(q.transpose(0, 2, 1, 3), log_pos)
        q_nope, q_pe = mx.split(q, [self.nope_dim], axis=-1)
        if self.use_rope:
            q_pe = rope(q_pe)

        if self.owns_kv:
            kv = self.kv_a_proj_with_mqa(x)
            kv_latent, k_pe = mx.split(kv, [self.kv_lora_rank], axis=-1)
            kv_latent = mx.expand_dims(self.kv_a_layernorm(kv_latent), axis=1)
            k_pe = k_pe.reshape(B, L, 1, self.rope_dim).transpose(0, 2, 1, 3)
            if self.use_rope:
                k_pe = rope(k_pe)
            if mla_cache is not None:
                kv_latent, k_pe = mla_cache.update_and_fetch(kv_latent, k_pe)
            shared_kv[self.layer_idx] = (kv_latent, k_pe)
        else:
            kv_latent, k_pe = shared_kv[self.kv_source]

        if has_indexer:
            topk = self.indexer(x, mask, rope, log_pos, index_cache)
        else:
            topk = prev_topk

        if topk is not None:
            if L == 1:
                idx = topk[:, :, 0, :, None]

                def gather(a):
                    shape = idx.shape[:-1] + (a.shape[-1],)
                    return mx.take_along_axis(a, mx.broadcast_to(idx, shape), axis=2)

                kv_latent = gather(kv_latent)
                if self.rope_dim:
                    k_pe = gather(k_pe)
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk, axis=-1)
            else:
                shape = topk.shape[:-1] + (kv_latent.shape[2],)
                sparse = mx.put_along_axis(
                    mx.zeros(shape, dtype=mx.bool_), topk, mx.array(True), axis=-1
                )
                mask = sparse if mask is None else sparse & mask

        if L == 1:
            # Decode attends to the latent directly.
            if self.rope_dim:
                pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
                if mask is not None:
                    min_value = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
                    pe_scores = mx.where(mask, pe_scores, min_value)
                mask = pe_scores
            output = scaled_dot_product_attention(
                self.embed_q(q_nope),
                kv_latent,
                kv_latent,
                cache=None,
                scale=self.scale,
                mask=mask,
            )
            output = self.unembed_out(output)
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
            if self.rope_dim:
                k_pe = mx.broadcast_to(k_pe, k.shape[:-1] + (self.rope_dim,))
                q = mx.concatenate([q_nope, q_pe], axis=-1)
                k = mx.concatenate([k, k_pe], axis=-1)
            else:
                q = q_nope
            output = scaled_dot_product_attention(
                q, k, v, cache=None, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if cache is not None and cache.caches:
            # Evaluate unread cache arrays, or long generations exhaust Metal buffers.
            state = [a for c in cache.caches for a in c.state]
            output = mx.depends(output, [a for a in state if isinstance(a, mx.array)])
        return self.o_proj(output), topk


class MoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.shared_experts = DeepseekV3MLP(
            args, intermediate_size=args.moe_intermediate_size * args.n_shared_experts
        )
        self.sharding_group = None

    def __call__(self, x):
        args = self.args
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, scores = mistral4_expert_select(
            self.gate(x.astype(mx.float32)),
            args.num_experts_per_tok,
            args.n_group,
            args.topk_group,
            args.routed_scaling_factor,
            True,
        )
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_experts(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = DeepseekV3MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask, rope, log_pos, cache, shared_kv, prev_topk):
        r, topk = self.self_attn(
            self.input_layernorm(x), mask, rope, log_pos, cache, shared_kv, prev_topk
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk


class MistralLarge4TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rope = LeRoPE(args.qk_rope_head_dim, args.rope_parameters)
        # KV source layers whose latent rows cross into or out of this stage.
        self.recv_sources = []
        self.send_sources = []

    def pipeline(self, group, split=None):
        if split is None:
            split = self._stage_sizes(group.size())[::-1]
        super().pipeline(group, split)
        start, end = self.start_idx, self.end_idx
        if self.args.indexer_types[start] != "full":
            raise ValueError(
                f"A pipeline stage cannot start at layer {start}, "
                "which reuses the top-k of the layer before it."
            )
        sources = self.args.kv_source_layers
        self.recv_sources = sorted({s for c, s in enumerate(sources) if s < start <= c})
        self.send_sources = sorted({s for c, s in enumerate(sources) if s < end <= c})

    def _stage_sizes(self, n):
        # Even stages, each moved to start at a layer that runs its own indexer.
        num_layers, kinds = len(self.layers), self.args.indexer_types
        bounds = [0]
        for i in range(1, n):
            b = max(round(i * num_layers / n), bounds[-1] + 1)
            while b < num_layers and kinds[b] != "full":
                b += 1
            bounds.append(b)
        bounds.append(num_layers)
        return [e - s for s, e in zip(bounds, bounds[1:])]

    def _latent_width(self, layer):
        rope = self.args.qk_rope_head_dim if self.args.no_rope_layers[layer] else 0
        return self.args.kv_lora_rank + rope

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)
        # Every cache of a layer has the same offset.
        first = cache[0][0] if cache[0] is not None else None
        offset = first.offset if first is not None else 0
        mask = create_attention_mask(h, first, return_array=True)

        dims, freqs = self.rope.dims, self.rope.freqs()

        def rope(x):
            return mx.fast.rope(
                x,
                dims,
                traditional=True,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=freqs,
            )

        B, L, D = h.shape
        log_pos = _log_positions(offset, L)
        rank, size = self.pipeline_rank, self.pipeline_size
        shared_kv = {}

        # Receive the hidden states, and the new latent rows of KV source layers
        # that run on an earlier stage. A replica cache holds those rows.
        if rank < size - 1:
            widths = [self._latent_width(s) for s in self.recv_sources]
            x = mx.distributed.recv_like(
                mx.zeros((B, L, D + sum(widths)), h.dtype), rank + 1
            )
            cuts = [D + sum(widths[:i]) for i in range(len(widths))]
            h, *rows = mx.split(x, cuts, axis=-1)
            replicas = []
            if cache[0] is not None and rows:
                replicas = cache[0].caches[-len(rows) :]
            for i, (s, r) in enumerate(zip(self.recv_sources, rows)):
                latent, pe = mx.split(r[:, None], [self.args.kv_lora_rank], axis=-1)
                if replicas:
                    latent, pe = replicas[i].update_and_fetch(latent, pe)
                shared_kv[s] = (latent, pe)

        topk = None
        for layer, c in zip(layers, cache):
            h, topk = layer(h, mask, rope, log_pos, c, shared_kv, topk)

        if rank > 0:
            # The new rows are the last L rows of the full latent.
            rows = [
                mx.concatenate(shared_kv[s], axis=-1)[:, 0, -L:]
                for s in self.send_sources
            ]
            sent = mx.distributed.send(mx.concatenate([h, *rows], axis=-1), rank - 1)
            h = mx.depends(h, [sent])
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, [sent])

        # Every rank needs the output of rank 0, which runs the last layers.
        if size > 1:
            h = mx.distributed.all_gather(h)[:B]
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text_args = TextArgs.from_dict(args.text_config)
        self.model = MistralLarge4TextModel(self.text_args)
        self.lm_head = nn.Linear(
            self.text_args.hidden_size, self.text_args.vocab_size, bias=False
        )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.lm_head(self.model(inputs, cache, input_embeddings))
        if cap := self.text_args.final_logit_softcapping:
            out = mx.tanh(out / cap) * cap
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            # Heads are sharded. The latent cache and the indexer are replicated,
            # so every rank selects the same top-k.
            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.num_heads //= N
            sh = group.rank() * attn.num_heads
            eh = sh + attn.num_heads

            def shard_heads(w):
                return w[sh:eh]

            attn.embed_q.apply(shard_heads)
            attn.unembed_out.apply(shard_heads)
            attn.softmax_scaling.apply(shard_heads)

            mlp = layer.mlp
            if isinstance(mlp, MoE):
                # Shard in place: the MoE aggregates the partial sums itself.
                mlp.sharding_group = group
                for m in (mlp.shared_experts, mlp.switch_mlp):
                    shard_inplace(m.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(m.up_proj, "all-to-sharded", group=group)
                    shard_inplace(m.down_proj, "sharded-to-all", group=group)
            else:
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

    def make_cache(self):
        caches = []
        for layer in self.layers:
            attn = layer.self_attn
            parts = []
            if attn.owns_kv:
                parts.append(KVCache())
            if attn.indexer is not None:
                parts.append(KVCache())
            caches.append(CacheList(*parts))
        # The first layer of a pipeline stage also holds the replica caches of
        # KV source layers that run on an earlier stage.
        if self.model.recv_sources:
            replicas = [KVCache() for _ in self.model.recv_sources]
            caches[0] = CacheList(*caches[0].caches, *replicas)
        return caches

    def sanitize(self, weights):
        args = self.text_args
        renames = [
            (r"^tok_embeddings\.", "model.embed_tokens."),
            (r"^norm\.", "model.norm."),
            (r"^output\.", "lm_head."),
            (r"^lerope\.alpha$", "model.rope.alpha"),
            (r"^layers\.", "model.layers."),
            (r"\.attention_norm\.", ".input_layernorm."),
            (r"\.ffn_norm\.", ".post_attention_layernorm."),
            (r"\.ssmax_scaling\.s$", ".softmax_scaling.scale"),
            (r"\.ssmax_scaling\.b$", ".softmax_scaling.bias"),
            (r"\.attention\.indexer\.", ".self_attn.indexer."),
            (r"\.attention\.softmax_scaling\.", ".self_attn.softmax_scaling."),
            (r"\.attention\.wq\.", ".self_attn.q_proj."),
            (r"\.attention\.wkv_a_with_mqa\.", ".self_attn.kv_a_proj_with_mqa."),
            (r"\.attention\.kv_a_norm\.", ".self_attn.kv_a_layernorm."),
            (r"\.attention\.wkv_b\.", ".self_attn.kv_b_proj."),
            (r"\.attention\.wo\.", ".self_attn.o_proj."),
            (r"\.feed_forward\.", ".mlp."),
            (r"\.shared_experts\.", ".mlp.shared_experts."),
            (r"\.experts\.", ".mlp.experts."),
            (r"\.gate\.weight$", ".mlp.gate.weight"),
            (r"\.w1\.", ".gate_proj."),
            (r"\.w2\.", ".down_proj."),
            (r"\.w3\.", ".up_proj."),
            (r"\.qscale_weight$", ".weight_scale_inv"),
        ]
        skip = ("vision_encoder.", "patch_merger.", "pre_mm_projector_norm.")
        skip += ("vision_language_adapter.", "mtp_dspark.")

        new_weights = {}
        for k, v in weights.items():
            # Converted weights already have MLX names.
            if not k.startswith(("model.", "lm_head.")):
                if k.startswith(skip):
                    continue
                layer = re.match(r"layers\.(\d+)\.", k)
                if layer and int(layer.group(1)) >= args.num_hidden_layers:
                    continue
                for pattern, repl in renames:
                    k = re.sub(pattern, repl, k)
            new_weights[k] = v
        weights = new_weights

        # The FP8 layers stay FP8, as mxfp8. The quantization_config tells the
        # FP8 checkpoint, and a convert without -q, from other mxfp8 builds.
        fp8 = (self.args.quantization_config or {}).get("quant_method") == "fp8"
        scale_keys = [k for k in weights if k.endswith(".weight_scale_inv")]
        if scale_keys and not fp8:
            raise ValueError(
                "FP8 weights need quant_method fp8 in quantization_config."
            )
        exact = []
        for k in scale_keys:
            wk = k.removesuffix("_scale_inv")
            scale = weights.pop(k)
            exact.append(mx.all(mx.power(2.0, mx.round(mx.log2(scale))) == scale))
            weights[wk], weights[f"{wk.removesuffix('weight')}scales"] = _fp8_as_mxfp8(
                weights[wk], scale
            )
        if exact and not mx.all(mx.stack(exact)).item():
            raise ValueError("FP8 block scales must be powers of two to run as mxfp8.")

        for l in range(args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales"]:
                    keys = [
                        f"{prefix}.mlp.experts.{e}.{m}.{k}"
                        for e in range(args.n_routed_experts)
                    ]
                    # A pipeline stage can get some experts of a layer that it
                    # does not run, from a file shared with its own layers.
                    if all(key in weights for key in keys):
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                            [weights.pop(key) for key in keys]
                        )

            attn = f"{prefix}.self_attn"
            if f"{attn}.kv_b_proj.weight" in weights:
                nope = args.qk_nope_head_dim
                if not args.no_rope_layers[l]:
                    nope += args.qk_rope_head_dim
                w = weights.pop(f"{attn}.kv_b_proj.weight")
                w = w.reshape(args.num_attention_heads, nope + args.v_head_dim, -1)
                weights[f"{attn}.embed_q.weight"] = mx.contiguous(
                    w[:, :nope, :].swapaxes(-1, -2)
                )
                weights[f"{attn}.unembed_out.weight"] = mx.contiguous(w[:, nope:, :])

        if fp8:
            layers = []
            for p, m in self.named_modules():
                if f"{p}.scales" not in weights:
                    continue
                if isinstance(m, SwitchLinear):
                    layer = Fp8SwitchLinear(m.input_dims, m.output_dims, m.num_experts)
                    layers.append((p, layer))
                elif isinstance(m, nn.Linear):
                    out_dims, in_dims = m.weight.shape
                    layers.append((p, Fp8Linear(in_dims, out_dims)))
            self.update_modules(tree_unflatten(layers))

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The checkpoint keeps these unquantized, so they stay in bf16.
            return not path.endswith(("mlp.gate", "indexer.wk", "indexer.weights_proj"))

        return predicate
