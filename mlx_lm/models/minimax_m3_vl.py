# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import SwigluOAI
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


class GemmaRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def _extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, weight=1.0 + self.weight, eps=self.eps)


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
    rope_parameters: Optional[dict] = None
    head_dim: int = 128
    rotary_dim: Optional[int] = None
    partial_rotary_factor: float = 0.5
    vocab_size: int = 200064
    tie_word_embeddings: bool = False
    routed_scaling_factor: float = 2.0
    use_qk_norm: bool = True
    use_gemma_norm: bool = True
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    mlp_layer_types: Optional[List[str]] = None
    moe_layer_freq: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    sparse_attention_config: Optional[dict] = None
    sparse_attention_freq: Optional[List[int]] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_block_size: Optional[int] = None
    index_topk_blocks: Optional[int] = None
    index_local_blocks: int = 1

    def __post_init__(self):
        # The published config nests these; a 5.x re-save spells them flat.
        sac = self.sparse_attention_config or {}
        if sac.get("use_sparse_attention"):
            self.index_n_heads = sac["sparse_num_index_heads"]
            self.index_head_dim = sac["sparse_index_dim"]
            self.index_block_size = sac["sparse_block_size"]
            self.index_topk_blocks = sac["sparse_topk_blocks"]
            self.index_local_blocks = sac.get(
                "sparse_local_block", self.index_local_blocks
            )
            self.sparse_attention_freq = sac.get(
                "sparse_attention_freq", self.sparse_attention_freq
            )
        rope = self.rope_parameters or {}
        self.rope_theta = rope.get("rope_theta", self.rope_theta)
        self.partial_rotary_factor = rope.get(
            "partial_rotary_factor", self.partial_rotary_factor
        )
        if self.rotary_dim is None:
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    def is_moe(self, layer_idx: int) -> bool:
        # transformers 5.x renamed moe_layer_freq to mlp_layer_types. The published
        # config carries only the old key, a re-save only the new one.
        if self.mlp_layer_types is not None:
            return self.mlp_layer_types[layer_idx] == "sparse"
        if self.moe_layer_freq is not None:
            return bool(self.moe_layer_freq[layer_idx])
        return True

    def is_sparse_attn(self, layer_idx: int) -> bool:
        if self.index_block_size is None:
            return False
        if self.layer_types is not None:
            return self.layer_types[layer_idx] != "full_attention"
        if self.sparse_attention_freq is not None:
            return bool(self.sparse_attention_freq[layer_idx])
        return True


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    model_type: str = "minimax_m3_vl"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextArgs.from_dict(self.text_config)


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, *, activation):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        x_up = self.up_proj(x)
        x_gate = self.gate_proj(x)
        return self.down_proj(self.activation(x_up, x_gate))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

        activation = SwigluOAI(alpha=args.swiglu_alpha, limit=args.swiglu_limit)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=activation,
        )

        self.shared_experts = DenseMLP(
            args.hidden_size,
            args.shared_intermediate_size,
            activation=activation,
        )

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        ne = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        shared_out = self.shared_experts(x_flat)

        # The reference routes in the weight dtype and upcasts only the logits.
        gates = self.gate(x_flat)
        scores = mx.sigmoid(gates.astype(mx.float32))
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x_flat.dtype)

        y = self.switch_mlp(x_flat, inds)
        y = (y * scores[..., None]).sum(axis=-2) * self.routed_scaling_factor
        y = y + shared_out

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y.reshape(*ne, -1)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        NormClass = GemmaRMSNorm if args.use_gemma_norm else nn.RMSNorm

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
            self.q_norm = NormClass(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = NormClass(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            dims=args.rotary_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
        )

        # The indexer picks this layer's key blocks; its weights sit on self_attn.
        self.is_sparse_attn = args.is_sparse_attn(layer_idx)
        if self.is_sparse_attn:
            self.index_n_heads = args.index_n_heads
            self.index_head_dim = args.index_head_dim
            self.index_block_size = args.index_block_size
            self.index_topk_blocks = args.index_topk_blocks
            self.index_local_blocks = args.index_local_blocks
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_n_heads * self.index_head_dim, bias=False
            )
            self.index_k_proj = nn.Linear(
                args.hidden_size, self.index_head_dim, bias=False
            )
            self.index_q_norm = NormClass(self.index_head_dim, eps=args.rms_norm_eps)
            self.index_k_norm = NormClass(self.index_head_dim, eps=args.rms_norm_eps)
            # The reference truncates cos/sin to the index head.
            self.index_rope = initialize_rope(
                dims=min(args.rotary_dim, self.index_head_dim),
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_parameters,
                max_position_embeddings=args.max_position_embeddings,
            )

    def _block_mask(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        offset: int,
        cache: Optional[Any],
    ) -> Optional[mx.array]:
        """Bool mask keeping only the top-k key blocks per query."""
        B, L, _ = x.shape
        H, D = self.index_n_heads, self.index_head_dim

        # Cache the index keys even when no block is dropped: later steps need them.
        k = self.index_k_proj(x).reshape(B, L, 1, D)
        k = self.index_k_norm(k).transpose(0, 2, 1, 3)
        k = self.index_rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), k.dtype))

        blk = self.index_block_size
        topk = self.index_topk_blocks
        S = k.shape[2]
        n_blk = (S + blk - 1) // blk
        # Every block fits in the budget, so the selection would keep all of them.
        if topk >= n_blk:
            return mask

        q = self.index_q_proj(x).reshape(B, L, H, D)
        q = self.index_q_norm(q).transpose(0, 2, 1, 3)
        q = self.index_rope(q, offset=offset)

        # Blocks are cut over cache slots, so place the queries by slot, not by
        # the per-row content offset the rope uses.
        qpos = mx.arange(S - L, S)
        causal = mx.arange(S)[None] <= qpos[:, None]
        scores = q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
        # A pad or future key must not win a block, so mask before pooling.
        per_key = mask if isinstance(mask, mx.array) else causal
        scores = mx.where(per_key, scores, -mx.inf)
        pad = n_blk * blk - S
        if pad:
            scores = mx.pad(scores, [(0, 0)] * 3 + [(0, pad)], constant_values=-mx.inf)
        # Score a block by its best key, then always keep the newest local blocks.
        block_scores = scores.reshape(B, H, L, n_blk, blk).max(axis=-1)
        if self.index_local_blocks > 0:
            qblk = (qpos // blk)[:, None]
            ids = mx.arange(n_blk)
            is_local = (ids <= qblk) & (ids > qblk - self.index_local_blocks)
            block_scores = mx.where(is_local, mx.inf, block_scores)

        inds = mx.argpartition(-block_scores, kth=topk - 1, axis=-1)[..., :topk]
        keep = mx.put_along_axis(
            mx.zeros(block_scores.shape, dtype=mx.bool_),
            inds,
            mx.array(True),
            axis=-1,
        )
        # A kept block still holds future keys, so mask again.
        keep = mx.repeat(keep, blk, axis=-1)[..., :S] & per_key
        # One selection per KV group, so widen it to every query head in the group.
        return mx.repeat(keep, self.num_attention_heads // H, axis=1)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        kv_cache = cache[0] if cache is not None else None
        offset = kv_cache.offset if kv_cache is not None else 0

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Select before the cache advances: it moves its offset array in place.
        if self.is_sparse_attn:
            mask = self._block_mask(
                x, mask, offset, cache[1] if cache is not None else None
            )
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
            # Keep the indexer cache in the graph: below the block budget nothing
            # consumes it, and the deferred updates pile up for the whole decode.
            if self.is_sparse_attn:
                kv_cache.keys = mx.depends(kv_cache.keys, cache[1].keys)

        out = scaled_dot_product_attention(
            q, k, v, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)

        self.is_moe = args.is_moe(layer_idx)
        if self.is_moe:
            self.block_sparse_moe = SparseMoeBlock(args)
        else:
            activation = SwigluOAI(alpha=args.swiglu_alpha, limit=args.swiglu_limit)
            self.mlp = DenseMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                activation=activation,
            )

        NormClass = GemmaRMSNorm if args.use_gemma_norm else nn.RMSNorm
        self.input_layernorm = NormClass(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = NormClass(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        if self.is_moe:
            r = r + self.block_sparse_moe(self.post_attention_layernorm(r))
        else:
            r = r + self.mlp(self.post_attention_layernorm(r))
        return r


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        NormClass = GemmaRMSNorm if args.use_gemma_norm else nn.RMSNorm
        self.norm = NormClass(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0][0] if cache[0] else None)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            h = layer(h, mask, c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.args = config
        self.model = TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
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

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        return self.language_model(inputs, cache)

    def make_cache(self) -> List[Any]:
        # Only sparse layers have indexer keys; an unused KVCache breaks .state.
        return [
            (
                CacheList(KVCache(), KVCache())
                if layer.self_attn.is_sparse_attn
                else CacheList(KVCache())
            )
            for layer in self.layers
        ]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        args = self.args.text_config
        # The vision tower and the projectors are not used.
        skip = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
        out = {k: v for k, v in weights.items() if not k.startswith(skip)}
        if args.tie_word_embeddings:
            out.pop("language_model.lm_head.weight", None)

        experts = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))
        for i in range(args.num_hidden_layers):
            moe = f"language_model.model.layers.{i}.block_sparse_moe"
            # Some exports keep a dense layer's MLP under block_sparse_moe. Its
            # gate is named gate_proj, the router's is gate, so this is exact.
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if f"{moe}.{proj}.weight" not in out:
                    continue
                mlp = f"language_model.model.layers.{i}.mlp"
                for part in ("weight", "scales", "biases"):
                    v = out.pop(f"{moe}.{proj}.{part}", None)
                    if v is not None:
                        out[f"{mlp}.{proj}.{part}"] = v

            # Some conversions fold the shared expert in as one more expert.
            # The expert axis is not the packed one, so the split is exact
            # on quantized weights.
            n = args.num_local_experts
            for proj in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
                w = out.get(f"{moe}.switch_mlp.{proj}.weight")
                if w is None or w.shape[0] != n + 1:
                    continue
                for part in ("weight", "scales", "biases"):
                    v = out.pop(f"{moe}.switch_mlp.{proj}.{part}", None)
                    if v is None:
                        continue
                    out[f"{moe}.switch_mlp.{proj}.{part}"] = mx.contiguous(v[:n])
                    out[f"{moe}.shared_experts.{proj}.{part}"] = mx.contiguous(v[n])

            for owner in (f"{moe}.switch_mlp", f"{moe}.shared_experts"):
                for part in ("weight", "scales", "biases"):
                    fused = out.pop(f"{owner}.gate_up_proj.{part}", None)
                    if fused is None:
                        continue
                    gate, up = mx.split(fused, 2, axis=-2)
                    out[f"{owner}.gate_proj.{part}"] = mx.contiguous(gate)
                    out[f"{owner}.up_proj.{part}"] = mx.contiguous(up)

            if f"{moe}.experts.0.w1.weight" not in out:
                continue
            for src, dst in experts:
                for part in ("weight", "scales", "biases"):
                    if f"{moe}.experts.0.{src}.{part}" not in out:
                        continue
                    out[f"{moe}.switch_mlp.{dst}.{part}"] = mx.stack(
                        [
                            out.pop(f"{moe}.experts.{e}.{src}.{part}")
                            for e in range(args.num_local_experts)
                        ]
                    )

        return out

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        # pipeline() blanks the layers it does not hold, so take the live ones.
        for layer in self.layers:
            attn = layer.self_attn
            # Check before the first mutation, or a failure leaves the layer
            # half sharded.
            heads = [attn.num_attention_heads, attn.num_key_value_heads]
            if attn.is_sparse_attn:
                heads.append(attn.index_n_heads)
            assert all(h % N == 0 for h in heads), (
                f"head counts {heads} are not all divisible by {N} ranks; "
                "is the model already sharded?"
            )
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.num_attention_heads //= N
            attn.num_key_value_heads //= N
            if attn.is_sparse_attn:
                # Each index head serves one KV group, so it follows the query heads.
                attn.index_q_proj = shard_linear(
                    attn.index_q_proj, "all-to-sharded", group=group
                )
                attn.index_n_heads //= N

            if layer.is_moe:
                moe = layer.block_sparse_moe
                # In place, so the block aggregates both experts with one sum.
                for mlp in (moe.switch_mlp, moe.shared_experts):
                    shard_inplace(mlp.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(mlp.up_proj, "all-to-sharded", group=group)
                    shard_inplace(mlp.down_proj, "sharded-to-all", group=group)
                moe.sharding_group = group
            else:
                mlp = layer.mlp
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def layers(self) -> List[nn.Module]:
        return self.language_model.model.pipeline_layers

    @property
    def model(self) -> nn.Module:
        # sharded_load() reaches for model.model.pipeline().
        return self.language_model.model

    def pipeline(self, group: mx.distributed.Group):
        self.language_model.model.pipeline(group)

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], Any]:
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
