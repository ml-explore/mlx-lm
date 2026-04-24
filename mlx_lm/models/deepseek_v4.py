# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import RotatingKVCache
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")

@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
    ):
        super().__init__()
        self.dims = dims

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
        inverse: bool = False,
        positions: Optional[mx.array] = None,
    ):
        dtype = x.dtype
        L = x.shape[-2]
        pos = (
            mx.arange(offset, offset + L, dtype=mx.float32)
            if positions is None
            else positions.astype(mx.float32)
        )
        freqs = pos[:, None] * self.inv_freq[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        if inverse:
            sin = -sin

        broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
        cos = cos.reshape(broadcast_shape).astype(dtype)
        sin = sin.reshape(broadcast_shape).astype(dtype)

        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        return out.reshape(*out.shape[:-2], out.shape[-2] * 2)


def _apply_partial_rope(
    x: mx.array,
    rope: DeepseekV4RoPE,
    offset: int = 0,
    inverse: bool = False,
    positions: Optional[mx.array] = None,
) -> mx.array:
    rope_dim = rope.dims
    if x.shape[-1] == rope_dim:
        return rope(x, offset=offset, inverse=inverse, positions=positions)
    nope, pe = mx.split(x, [x.shape[-1] - rope_dim], axis=-1)
    pe = rope(pe, offset=offset, inverse=inverse, positions=positions)
    return mx.concatenate([nope, pe], axis=-1)

@mx.compile
def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


class HyperConnection(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def compute_weights(self, x: mx.array):
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D).astype(mx.float32)
        rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (flat @ self.fn.T) * rsqrt
        return hc_split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )

    def collapse(self, x: mx.array):
        pre, post, comb = self.compute_weights(x)
        collapsed = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return collapsed.astype(x.dtype), post, comb

    def expand(
        self,
        block_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ):
        y = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
        y = y + mx.einsum(
            "bsij,bsjd->bsid",
            comb.astype(mx.float32),
            residual.astype(mx.float32),
        )
        return y.astype(block_out.dtype)


class HyperHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D).astype(mx.float32)
        rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (flat @ self.fn.T) * rsqrt
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        flat = x.reshape(-1, self.hidden_dim)
        logits = flat.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        scores = _score_func(logits, self.scoring_func)

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds = self.tid2eid[input_ids.reshape(-1)].astype(mx.int32)
        else:
            biased = scores + self.e_score_correction_bias
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]

        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.scoring_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        route_shape = (*x.shape[:-1], self.top_k)
        inds = inds.reshape(route_shape)
        weights = weights.reshape(route_shape)
        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class DeepseekV4Cache:
    def __init__(self, sliding_window: int):
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        self.compressor_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}
        self.indexer_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def state(self):
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            tuple(
                self.compressor_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")
            ),
            tuple(
                self.indexer_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")
            ),
        )

    @state.setter
    def state(self, value):
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self.compressor_state = dict(
            zip(("buffer_kv", "buffer_gate", "pooled"), compressor_state)
        )
        self.indexer_state = dict(
            zip(("buffer_kv", "buffer_gate", "pooled"), indexer_state)
        )

    @property
    def meta_state(self):
        return self.local.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    def make_mask(self, *args, **kwargs):
        return self.local.make_mask(*args, **kwargs)

    def is_trimmable(self):
        return self.local.is_trimmable()

    def trim(self, n):
        return self.local.trim(n)

    def size(self):
        return self.local.size()

    def empty(self):
        return self.local.empty()

    @property
    def nbytes(self):
        total = self.local.nbytes
        for state in (self.compressor_state, self.indexer_state):
            for value in state.values():
                if value is not None:
                    total += value.nbytes
        return total

    def _branch_state(self, state_key: str):
        return (
            self.indexer_state
            if state_key == "indexer_state"
            else self.compressor_state
        )

    def accumulate_windows(
        self,
        kv: mx.array,
        gate: mx.array,
        state_key: str,
        ratio: int,
        start_pos: int,
    ):
        state = self._branch_state(state_key)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = mx.concatenate([buf_kv, kv], axis=1)
            gate = mx.concatenate([buf_gate, gate], axis=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def update_pool(self, new_pooled: mx.array, state_key: str) -> mx.array:
        state = self._branch_state(state_key)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = (
                new_pooled
                if pool is None
                else mx.concatenate([pool, new_pooled], axis=1)
            )
            state["pooled"] = pool
        if pool is None:
            pool = mx.zeros(
                (new_pooled.shape[0], 0, new_pooled.shape[-1]), new_pooled.dtype
            )
        return pool


class Compressor(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def _overlap_transform(self, x: mx.array, fill_value: float):
        B, W, R, _ = x.shape
        out = mx.full((B, W, 2 * R, self.head_dim), fill_value, dtype=x.dtype)
        out[:, :, R:] = x[:, :, :, self.head_dim :]
        out[:, 1:, :R] = x[:, :-1, :, : self.head_dim]
        return out

    def __call__(
        self,
        x: mx.array,
        rope: DeepseekV4RoPE,
        cache: Optional[DeepseekV4Cache],
        start_pos: int,
        state_key: str = "compressor_state",
    ) -> mx.array:
        B, _, _ = x.shape
        kv = self.wkv(x)
        gate = self.wgate(x)
        if cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = start_pos
        else:
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, state_key, self.compress_ratio, start_pos
            )

        if ready_kv.shape[1] == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            W = ready_kv.shape[1] // self.compress_ratio
            kv = ready_kv.reshape(B, W, self.compress_ratio, self.out_dim)
            gate = ready_gate.reshape(
                B, W, self.compress_ratio, self.out_dim
            ) + self.ape.astype(ready_gate.dtype)
            if self.overlap:
                kv = self._overlap_transform(kv, 0.0)
                gate = self._overlap_transform(gate, -float("inf"))
            weights = mx.softmax(gate.astype(mx.float32), axis=2, precise=True).astype(
                kv.dtype
            )
            new_pooled = (kv * weights).sum(axis=2)
            new_pooled = self.norm(new_pooled.astype(x.dtype))
            positions = (
                mx.arange(new_pooled.shape[1], dtype=mx.float32) * self.compress_ratio
                + pool_base
            )
            new_pooled = _apply_partial_rope(
                new_pooled[:, None], rope, positions=positions
            ).squeeze(1)

        if cache is not None:
            return cache.update_pool(new_pooled, state_key)
        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        rope: DeepseekV4RoPE,
        position_rope: DeepseekV4RoPE,
        cache: Optional[DeepseekV4Cache],
        start_pos: int,
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, rope, cache, start_pos, state_key="indexer_state")
        if pooled.shape[1] == 0:
            return None

        offset = start_pos
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, position_rope, offset)

        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        k = min(self.index_topk, pooled.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


class V4Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        rope_theta = (
            config.compress_rope_theta if self.compress_ratio else config.rope_theta
        )
        rope_scaling = config.rope_scaling if self.compress_ratio else None
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            rope_theta,
            rope_scaling,
            config.max_position_embeddings,
        )
        self.compress_rope = self.rope
        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.compress_ratio)

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        B, L = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.o_groups
        out = out.reshape(B, L, self.o_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            pieces = []
            for group_idx in range(self.o_groups):
                rows = slice(
                    group_idx * self.o_lora_rank,
                    (group_idx + 1) * self.o_lora_rank,
                )
                biases = (
                    self.wo_a.biases[rows] if self.wo_a.biases is not None else None
                )
                y = mx.quantized_matmul(
                    out[:, :, group_idx, :],
                    self.wo_a.weight[rows],
                    scales=self.wo_a.scales[rows],
                    biases=biases,
                    transpose=True,
                    group_size=self.wo_a.group_size,
                    bits=self.wo_a.bits,
                    mode=self.wo_a.mode,
                )
                if "bias" in self.wo_a:
                    y = y + self.wo_a.bias[rows]
                pieces.append(y)
            return mx.concatenate(pieces, axis=-1)

        weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, weight)
        out = out.reshape(B, L, self.o_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache
        if isinstance(cache, DeepseekV4Cache):
            local_cache = cache

        offset = local_cache.offset if local_cache is not None else 0
        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q * mx.rsqrt(
            (q.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)
            + self.config.rms_norm_eps
        )
        q = q.astype(x.dtype)
        q = q.transpose(0, 2, 1, 3)
        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim)
        kv = kv.transpose(0, 2, 1, 3)

        q = _apply_partial_rope(q, self.rope, offset)
        kv = _apply_partial_rope(kv, self.rope, offset)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv

        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
            if hasattr(self, "indexer") and pooled.shape[1] > 0:
                topk = self.indexer(
                    x, q_residual, self.compress_rope, self.rope, v4_cache, offset
                )
                if topk is not None:
                    expanded = mx.broadcast_to(
                        pooled[:, None, None, :, :],
                        (B, 1, L, pooled.shape[1], self.head_dim),
                    )
                    idx = topk[:, None, :, :, None]
                    pooled = mx.take_along_axis(
                        expanded,
                        mx.broadcast_to(idx, idx.shape[:-1] + (self.head_dim,)),
                        axis=3,
                    ).reshape(B, 1, -1, self.head_dim)
                else:
                    pooled = pooled[:, None]
            else:
                pooled = pooled[:, None]
            full_kv = mx.concatenate([full_kv, pooled], axis=2)

        if mask is not None and full_kv.shape[2] > mask.shape[-1]:
            pad = mx.ones(
                mask.shape[:-1] + (full_kv.shape[2] - mask.shape[-1],), dtype=mask.dtype
            )
            mask = mx.concatenate([mask, pad], axis=-1)

        out = scaled_dot_product_attention(
            q,
            full_kv,
            full_kv,
            cache=local_cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = V4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc.collapse(h)
        x = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        h = self.attn_hc.expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc.collapse(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return self.ffn_hc.expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache.local
            if isinstance(first_cache, DeepseekV4Cache)
            else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, DeepseekV4Cache):
                cache_item = cache_item.local
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith(
                (
                    ".ffn.switch_mlp.gate_proj",
                    ".ffn.switch_mlp.up_proj",
                    ".ffn.switch_mlp.down_proj",
                )
            ):
                return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                caches.append(DeepseekV4Cache(self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        def scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        def dequant_fp8(weight: mx.array, scale: mx.array, block_size: int = 128):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = scale_to_float(scale)
            m, n = weight.shape
            pad_m = (-m) % block_size
            pad_n = (-n) % block_size
            weight = mx.pad(weight, ((0, pad_m), (0, pad_n)))
            weight = weight.reshape(
                (m + pad_m) // block_size,
                block_size,
                (n + pad_n) // block_size,
                block_size,
            )
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_m, n + pad_n)
            return weight[:m, :n].astype(mx.bfloat16)

        def dequant_fp4(weight: mx.array, scale: mx.array, block_size: int = 32):
            table = mx.array(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=mx.float32,
            )
            packed = weight.astype(mx.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = mx.stack([mx.take(table, low), mx.take(table, high)], axis=-1)
            unpacked = unpacked.reshape(weight.shape[0], weight.shape[1] * 2)
            scale = mx.repeat(scale_to_float(scale), block_size, axis=-1)
            return (unpacked * scale).astype(mx.bfloat16)

        new_weights = {}
        for k, v in weights.items():
            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[wk] = dequant_fp4(weight, v)
            elif weight.dtype == mx.uint8:
                new_weights[wk] = dequant_fp8(weight, v)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stacked = [
                        weights.pop(f"{prefix}.{e}.{src}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.weight"] = (
                        mx.stack(stacked)
                    )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b, "all-to-sharded", group=group
            )
            layer.attn.wo_b = shard_linear(
                layer.attn.wo_b, "sharded-to-all", group=group
            )
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
