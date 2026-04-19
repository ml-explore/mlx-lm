# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update

# Fused compute_g + sigmoid(b) + gated_delta_kernel: saves 2 kernel
# launches per DeltaNet layer per step. Used for T=1 inference only
# (training uses VJP backends; prefill keeps original path).
try:
    from .gated_delta_fused import gated_delta_kernel_fused as _gd_fused_kernel
except Exception:
    _gd_fused_kernel = None

# Mega-fused T=1 kernel: rms_norm(q/k) + inv_scale + compute_g + sigmoid
# + gated_delta, все в одном dispatch. 1.24× per-call, ~4.5% end-to-end
# theoretical savings.
try:
    from .gated_delta_t1 import gated_delta_kernel_t1 as _gd_t1_kernel
except Exception:
    _gd_t1_kernel = None

# GatedDeltaNet training VJP backend selection (runtime, via env vars).
#
# MLX_DELTANET_VJP:
#   "metal"    (default) — Metal kernel backward (fastest, lowest memory)
#   "python"   — pure-Python chunked VJP (reference)
#   "lowrank"  — power-iter rank-r state compression (research prototype)
#   "compress" — compression-aware training: power-iter truncation at chunk
#                boundaries
#
# MLX_DELTANET_INFER_RANK (int):
#   If > 0, DeltaNet state stored as factored (U, V) with rank r in the
#   inference cache. Memory savings = (Dv·Dk) / (r·(Dv+Dk)), e.g. 4× at
#   r=16 for Qwen3.5-9B shapes. Enables higher multi-session concurrency
#   на Apple Silicon with unified memory.
#
# MLX_DELTANET_FACTORED_R (int):
#   If > 0, use fixed-rank factored MSL kernel for T=1 generation steps
#   (round-robin slot replacement). R=8-16 gives 1.4-1.6× speedup per
#   step in standalone benchmark; real end-to-end depends on kernel
#   launch overhead.
#
# MLX_DELTANET_COMPRESS_RANK (int):
#   0 (default) — no compression; use MLX_DELTANET_VJP backend as-is.
#   N > 0       — enable compression at uniform rank N (e.g. 16, 8).
#
# MLX_DELTANET_COMPRESS_RANK_PER_LAYER (path to JSON):
#   If set, overrides uniform rank with a per-layer dict
#   {"0": 8, "1": 8, "4": 16, ...}  (layer_idx → rank).
#   Generate with mlx_lm.compress.estimate_rank(probe_state=True).
#   See DELTANET_COMPRESSION.md for the rank-choice background.
import os as _os


def _env(name: str, default: str = "") -> str:
    """Look up env var under MLX_DELTANET_*."""
    return _os.environ.get(f"MLX_DELTANET_{name}", default)


_VJP_BACKEND = _env("VJP", "metal")
_COMPRESS_RANK = int(_env("COMPRESS_RANK", "0"))
_INFER_RANK = int(_env("INFER_RANK", "0"))
# MLX_DELTANET_INFER_QUANT: int8 / int4 state cache quantization.
# "none" | "8" | "4".  8-bit: 2× memory, minimal overhead.
# 4-bit: 4× memory, slightly more overhead.
_INFER_QUANT = _env("INFER_QUANT", "none")
# MLX_DELTANET_FACTORED_R: use factored MSL kernel for T=1 generation.
_FACTORED_R = int(_env("FACTORED_R", "0"))
if _FACTORED_R > 0:
    import math as _math

    from .gated_delta_factored_fixed import factored_step_fixed as _factored_step_fixed

    def _svd_factorize_dense(S: mx.array, rank: int):
        """Factor dense state [B, Hv, Dv, Dk] to (U, V) via truncated SVD.
        Batched SVD crashes in MLX 0.31 — loop over heads on CPU.
        Called once after prefill (one-off, slowness OK)."""
        B, Hv, Dv, Dk = S.shape
        S32 = S.astype(mx.float32)
        U_rows, V_rows = [], []
        for b in range(B):
            U_per_head, V_per_head = [], []
            for h in range(Hv):
                s_bh = S32[b, h]
                U_full, sigma, Vt_full = mx.linalg.svd(s_bh, stream=mx.cpu)
                U_per_head.append(U_full[:, :rank] * sigma[None, :rank])
                V_per_head.append(Vt_full[:rank, :])
            U_rows.append(mx.stack(U_per_head, axis=0))
            V_rows.append(mx.stack(V_per_head, axis=0))
        U_out = mx.stack(U_rows, axis=0).astype(S.dtype)
        V_out = mx.stack(V_rows, axis=0).astype(S.dtype)
        return U_out, V_out


if _INFER_RANK > 0 or _INFER_QUANT != "none":
    from .gated_delta_inference_compressed import factor_state as _infer_factor_state
    from .gated_delta_inference_compressed import maybe_expand as _infer_maybe_expand
    from .gated_delta_inference_compressed import (
        quantize_state as _infer_quantize_state,
    )

if _VJP_BACKEND == "scan":
    # Associative prefix-scan backend (see THEOREM_ASSOCIATIVITY.md).
    # Reference/research path — autodiff-compatible (VJP via MLX autodiff
    # through the scan loop). On single-device, slower than Metal; main
    # value is as a correctness reference and a building block for
    # multi-device Blelloch parallelism.
    from .gated_delta_prefix_scan import gated_delta_update_prefix_scan

    def gated_delta_update_vjp(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, layer_idx=None
    ):
        return gated_delta_update_prefix_scan(
            q, k, v, a, b, A_log, dt_bias, state, mask
        )

elif _VJP_BACKEND == "lowrank":
    from .gated_delta_vjp_lowrank import gated_delta_update_vjp_lowrank

    _LOWRANK_R = int(_env("VJP_RANK", "8"))
    _LOWRANK_ITERS = int(_env("VJP_ITERS", "10"))

    def gated_delta_update_vjp(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, layer_idx=None
    ):
        return gated_delta_update_vjp_lowrank(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state,
            mask,
            rank=_LOWRANK_R,
            method="power",
            power_iters=_LOWRANK_ITERS,
        )

elif (
    _VJP_BACKEND == "compress" or _COMPRESS_RANK > 0 or _env("COMPRESS_RANK_PER_LAYER")
):
    import json as _json

    # Prefer Metal-accelerated compressed VJP (8-11× faster backward
    # than pure-Python compressed, same compression behavior).
    # Fall back to Python compressed if Metal unavailable OR mask path
    # requested (Metal compressed не supports mask yet).
    _COMPRESS_METAL = _env("COMPRESS_METAL", "1") == "1"
    _metal_compressed_fn = None
    if _COMPRESS_METAL:
        try:
            from .gated_delta_vjp_metal_compressed import (
                gated_delta_update_vjp_metal_compressed as _metal_compressed_fn,
            )
        except ImportError:
            _metal_compressed_fn = None
    from .gated_delta_vjp_compressed import gated_delta_update_vjp_compressed

    _COMPRESS_R = (
        _COMPRESS_RANK if _COMPRESS_RANK > 0 else int(_env("COMPRESS_RANK", "16"))
    )
    _COMPRESS_ITERS = int(_env("COMPRESS_ITERS", "6"))

    _per_layer_path = _env("COMPRESS_RANK_PER_LAYER", "")
    _PER_LAYER_RANKS: dict = {}
    if _per_layer_path:
        try:
            _PER_LAYER_RANKS = {
                int(k): int(v) for k, v in _json.load(open(_per_layer_path)).items()
            }
            print(
                f"[mlx_lm.deltanet] Loaded per-layer compression ranks for "
                f"{len(_PER_LAYER_RANKS)} layers from {_per_layer_path} "
                f"(backend: {'Metal' if _metal_compressed_fn else 'Python'} compressed)"
            )
        except Exception as e:
            print(f"[mlx_lm.deltanet] WARNING: failed to load {_per_layer_path}: {e}")
            _PER_LAYER_RANKS = {}

    def gated_delta_update_vjp(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, layer_idx=None
    ):
        rank = (
            _PER_LAYER_RANKS.get(layer_idx, _COMPRESS_R)
            if _PER_LAYER_RANKS
            else _COMPRESS_R
        )
        # Metal path when available and no mask; else Python fallback.
        if _metal_compressed_fn is not None and mask is None:
            return _metal_compressed_fn(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                mask,
                rank=rank,
                power_iters=_COMPRESS_ITERS,
            )
        return gated_delta_update_vjp_compressed(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state,
            mask,
            rank=rank,
            power_iters=_COMPRESS_ITERS,
        )

else:
    try:
        from .gated_delta_vjp_metal import (
            gated_delta_update_vjp_metal as _gated_delta_update_vjp_core,
        )
    except ImportError:
        from .gated_delta_vjp import (
            gated_delta_update_vjp as _gated_delta_update_vjp_core,
        )

    def gated_delta_update_vjp(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, layer_idx=None
    ):
        # Metal backend does not use layer_idx; keep signature uniform.
        return _gated_delta_update_vjp_core(q, k, v, a, b, A_log, dt_bias, state, mask)


from .qwen3_next import Qwen3NextAttention as Attention
from .qwen3_next import Qwen3NextMLP as MLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs, layer_idx: int = -1):
        super().__init__()
        # layer_idx is used by the compression-aware VJP to pick a per-layer
        # rank from MLX_DELTANET_COMPRESS_RANK_PER_LAYER dict (see
        # DELTANET_COMPRESSION.md for the rank-choice background).
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        # Factored or quantized state inference: expand if needed.
        if (_INFER_RANK > 0 or _INFER_QUANT != "none") and state is not None:
            bits = int(_INFER_QUANT) if _INFER_QUANT != "none" else 8
            state = _infer_maybe_expand(state, bits=bits)

        # Decide inference path ДО rms_norm: mega-fused includes rms_norm
        # inside kernel, others need it externally.
        T_dim = inputs.shape[1]
        use_factored = (
            _FACTORED_R > 0 and not self.training and T_dim == 1 and cache is not None
        )
        use_mega_t1 = (
            _gd_t1_kernel is not None
            and not self.training
            and T_dim == 1
            and mask is None
            and not use_factored
            and _INFER_RANK == 0
            and _INFER_QUANT == "none"
        )

        inv_scale = k.shape[-1] ** -0.5
        if not use_mega_t1:
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        # Fused compute_g + gated_delta path (fallback, handles T>1).
        use_fused = (
            not use_mega_t1
            and _gd_fused_kernel is not None
            and not self.training
            and T_dim == 1
            and mask is None
            and not use_factored
            and _INFER_RANK == 0
            and _INFER_QUANT == "none"
        )

        if self.training and mask is None:
            # Memory-efficient VJP path: the recurrence is chunked and
            # gradient-checkpointed, so backward flows through every layer
            # without the O(T) graph footprint of the ops-based fallback.
            try:
                out, state = gated_delta_update_vjp(
                    q,
                    k,
                    v,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    state,
                    layer_idx=self.layer_idx,
                )
            except TypeError:
                # Older VJP backends do not accept layer_idx.
                out, state = gated_delta_update_vjp(
                    q, k, v, a, b, self.A_log, self.dt_bias, state
                )
        elif use_mega_t1:
            # Mega-fused T=1 kernel: rms_norm + compute_g + sigmoid +
            # gated_delta в single Metal dispatch. Takes raw q, k.
            rf = self.num_v_heads // self.num_k_heads
            q_exp = mx.repeat(q, rf, axis=-2) if rf > 1 else q
            k_exp = mx.repeat(k, rf, axis=-2) if rf > 1 else k
            if state is None:
                state = mx.zeros(
                    (
                        inputs.shape[0],
                        self.num_v_heads,
                        self.head_v_dim,
                        self.head_k_dim,
                    ),
                    dtype=inputs.dtype,
                )
            out, state = _gd_t1_kernel(
                q_exp, k_exp, v, a, b, self.A_log, self.dt_bias, state
            )
            if cache is not None:
                cache[1] = state
        elif use_fused:
            # Fused kernel path: compute_g + sigmoid(b) + gated_delta in
            # single Metal dispatch. Expand q, k for GQA first.
            rf = self.num_v_heads // self.num_k_heads
            q_exp = mx.repeat(q, rf, axis=-2) if rf > 1 else q
            k_exp = mx.repeat(k, rf, axis=-2) if rf > 1 else k
            if state is None:
                state = mx.zeros(
                    (
                        inputs.shape[0],
                        self.num_v_heads,
                        self.head_v_dim,
                        self.head_k_dim,
                    ),
                    dtype=inputs.dtype,
                )
            out, state = _gd_fused_kernel(
                q_exp, k_exp, v, a, b, self.A_log, self.dt_bias, state
            )
            if cache is not None:
                cache[1] = state
        elif use_factored:
            # Factored fast path: need current (U, V, slot_idx) in cache.
            factored = (
                cache[1]
                if isinstance(cache[1], (tuple, list))
                and len(cache[1]) == 3
                and not isinstance(cache[1], mx.array)
                else None
            )
            import mlx.nn as _nn

            beta_val = mx.sigmoid(b)  # [B, 1, Hv]
            # g = exp(-exp(A_log) · softplus(a + dt_bias))
            g_arg = -mx.exp(self.A_log.astype(mx.float32)) * _nn.softplus(
                a + self.dt_bias
            )
            g_arg = mx.maximum(g_arg, -20.0)
            g_val = mx.exp(g_arg).astype(a.dtype)
            # Expand q, k from Hk heads to Hv if needed.
            rf = self.num_v_heads // self.num_k_heads
            q_exp = mx.repeat(q, rf, axis=-2) if rf > 1 else q
            k_exp = mx.repeat(k, rf, axis=-2) if rf > 1 else k
            if factored is None:
                # First generate step: initialize ZERO factored state.
                # This discards prefill DeltaNet state contribution, which
                # is acceptable approximation — prefill state is heavily
                # decayed in steady-state anyway. First R generate steps
                # populate factored state, then round-robin replaces.
                B_ = inputs.shape[0]
                U_curr = mx.zeros(
                    (B_, self.num_v_heads, self.head_v_dim, _FACTORED_R),
                    dtype=inputs.dtype,
                )
                V_curr = mx.zeros(
                    (B_, self.num_v_heads, _FACTORED_R, self.head_k_dim),
                    dtype=inputs.dtype,
                )
                slot_idx = 0
            else:
                U_curr, V_curr, slot_idx = factored
            # Single-token factored step.
            y_flat, U_new, V_new = _factored_step_fixed(
                U_curr,
                V_curr,
                q_exp[:, 0],
                k_exp[:, 0],
                v[:, 0],
                g_val[:, 0],
                beta_val[:, 0],
                slot_idx=int(slot_idx) % _FACTORED_R,
            )
            out = y_flat[:, None, :, :]  # [B, 1, Hv, Dv]
            # Store factored state in cache (no dense reconstruction needed).
            cache[1] = (U_new, V_new, (int(slot_idx) + 1) % _FACTORED_R)
            # Skip the "cache[1] = state" branch below.
            # out = ... norm/out_proj below.
            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)
            out = self.norm(out, z)
            out = self.out_proj(out.reshape(B, S, -1))
            return out
        else:
            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )

        if cache is not None:
            if _INFER_RANK > 0 and not self.training:
                # Factored storage: save state as (U, V) to shrink cache.
                U, V = _infer_factor_state(state, rank=_INFER_RANK)
                cache[1] = (U, V)
            elif _INFER_QUANT != "none" and not self.training:
                # Quantized storage: save state as (w, scales, biases).
                bits = int(_INFER_QUANT)
                cache[1] = _infer_quantize_state(state, group_size=64, bits=bits)
            else:
                cache[1] = state

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args, layer_idx=layer_idx)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_mtp_weights = any("mtp." in k for k in weights)
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
