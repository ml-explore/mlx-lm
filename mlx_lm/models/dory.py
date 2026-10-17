# MLX port of NVIDIA Dory (DoryForCausalLM): looped nGPT-style model. Written from vLLM PR #60149 / llama.cpp dory.cpp.
# Drop-in for mlx_lm.models (model_type "dory"). v2: fused elementwise blocks, sliding-window cache, last-token logits.
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "dory"
    vocab_size: int = 131072
    hidden_size: int = 2560
    intermediate_size: int = 7168
    num_hidden_layers: int = 72
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    sliding_window: int = 1024
    n_input_layers: int = 16
    n_recurrent_layers: int = 40
    n_output_layers: int = 16
    n_recurrent_loops: int = 2
    recurrent_kv_cache_mode: str = "per_loop"  # or "last_loop"
    ngpt_alpha_init: float = 0.05
    initializer_range: float = 0.0177
    attention_output_gate_logit_scale_init: Optional[float] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                (
                    "mlp"
                    if i % 2
                    else ("full_attention" if i % 8 == 6 else "sliding_attention")
                )
                for i in range(self.num_hidden_layers)
            ]
        rp = self.rope_parameters or {}
        self.theta_sliding = float(
            rp.get("sliding_attention", {}).get("rope_theta", 10000.0)
        )
        self.theta_full = float(
            rp.get("full_attention", {}).get("rope_theta", 10000000.0)
        )
        assert self.recurrent_kv_cache_mode in ("per_loop", "last_loop")


_EPS = 1e-6


def _jn32(xf):
    return xf * mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + _EPS)


@partial(mx.compile, shapeless=True)
def justnorm(x):
    return _jn32(x.astype(mx.float32)).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def slerp_norm(res, out, alpha):
    a = _jn32(res.astype(mx.float32))
    b = _jn32(out.astype(mx.float32))
    return _jn32(a + alpha * (b - a)).astype(res.dtype)


@partial(mx.compile, shapeless=True)
def loop_inject(state, input_h):
    return _jn32((state + input_h).astype(mx.float32)).astype(state.dtype)


@partial(mx.compile, shapeless=True)
def qk_prep(qk, scale):
    # unit-norm per head, q heads scaled by sqk^2, k heads by 1 (scale is (1, H+K, 1, d))
    return (_jn32(qk.astype(mx.float32)) * scale).astype(qk.dtype)


@partial(mx.compile, shapeless=True)
def gate_out(o, g, sg2):
    y = o.astype(mx.float32) * mx.sigmoid(g.astype(mx.float32) * sg2)
    return _jn32(y).astype(o.dtype)


@partial(mx.compile, shapeless=True)
def mlp_act(g, u, sg, su):
    gf = g.astype(mx.float32) * sg
    uf = u.astype(mx.float32) * su
    return _jn32(nn.silu(gf) * uf).astype(g.dtype)


class RingKVCache:
    """Sliding-window KV cache with an in-place ring buffer for decoding (slot = abs_position % window)."""

    def __init__(self, window):
        self.W = window
        self.keys = None
        self.values = None
        self.offset = 0

    def _temporal(self):
        if self.keys is None:
            return None, None
        if self.offset <= self.W:
            return self.keys, self.values
        shift = (self.offset - self.W) % self.W
        return mx.roll(self.keys, -shift, axis=2), mx.roll(self.values, -shift, axis=2)

    def update_and_fetch(self, k, v):
        T = k.shape[2]
        W = self.W
        if T == 1 and self.keys is not None and self.keys.shape[2] == W:
            slot = self.offset % W
            self.keys[..., slot : slot + 1, :] = k
            self.values[..., slot : slot + 1, :] = v
            self.offset += 1
            return (
                self.keys,
                self.values,
            )  # order is irrelevant for T=1 (keys carry their own RoPE)
        ok, ov = self._temporal()
        K = k if ok is None else mx.concatenate([ok, k], axis=2)
        V = v if ov is None else mx.concatenate([ov, v], axis=2)
        self.offset += T
        if K.shape[2] >= W:
            shift = (self.offset - W) % W
            self.keys = mx.contiguous(mx.roll(K[..., -W:, :], shift, axis=2))
            self.values = mx.contiguous(mx.roll(V[..., -W:, :], shift, axis=2))
        else:
            self.keys, self.values = K, V
        return K, V

    @property
    def state(self):
        return [] if self.keys is None else (self.keys, self.values)

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return False

    def size(self):
        return 0 if self.keys is None else self.keys.shape[2]

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes


class DoryAttention(nn.Module):
    def __init__(self, args: ModelArgs, sliding: bool):
        super().__init__()
        self.args = args
        self.sliding = sliding
        d, H, K = args.head_dim, args.num_attention_heads, args.num_key_value_heads
        # fused [q | k | v | g] projection (sanitize concatenates the checkpoint's separate weights)
        self.qkvg_proj = nn.Linear(args.hidden_size, (2 * H + 2 * K) * d, bias=False)
        self.o_proj = nn.Linear(H * d, args.hidden_size, bias=False)
        self.sqk = mx.ones((H * d,))
        self.s_gate = mx.ones((H * d,))
        theta = args.theta_sliding if sliding else args.theta_full
        self.rope = nn.RoPE(d, traditional=False, base=theta)
        std = args.initializer_range
        gate_init = args.attention_output_gate_logit_scale_init
        if gate_init is None:
            gate_init = math.sqrt(args.hidden_size)
        self._sqk_factor = math.sqrt(d) / std
        self._gate_factor = math.sqrt(gate_init) / std
        self._scales = None

    def _get_scales(self):
        if self._scales is None:
            H, K, d = (
                self.args.num_attention_heads,
                self.args.num_key_value_heads,
                self.args.head_dim,
            )
            sqk = self.sqk.astype(mx.float32) * self._sqk_factor
            sg = self.s_gate.astype(mx.float32) * self._gate_factor
            qk_scale = mx.concatenate(
                [
                    (sqk * sqk).reshape(1, H, 1, d),
                    mx.ones((1, K, 1, d), dtype=mx.float32),
                ],
                axis=1,
            )
            self._scales = (qk_scale, sg * sg)
            mx.eval(self._scales)
        return self._scales

    def __call__(self, x, cache=None, reset_to=None):
        B, T, _ = x.shape
        H, K, d = (
            self.args.num_attention_heads,
            self.args.num_key_value_heads,
            self.args.head_dim,
        )
        qk_scale, sg2 = self._get_scales()
        y = self.qkvg_proj(x)
        n = (H + K) * d
        qk = y[..., :n].reshape(B, T, H + K, d).transpose(0, 2, 1, 3)
        v = y[..., n : n + K * d].reshape(B, T, K, d).transpose(0, 2, 1, 3)
        g = y[..., n + K * d :]
        if cache is not None and reset_to is not None:
            cache.offset = reset_to  # last_loop: overwrite slots of the current chunk
        off = cache.offset if cache is not None else 0
        qk = qk_prep(
            self.rope(qk, offset=off), qk_scale
        )  # RoPE first, then nGPT qk norm
        q, k = qk[:, :H], qk[:, H:]
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        S = k.shape[2]
        W = self.args.sliding_window
        mask = None
        if self.sliding:
            if T > 1 or S > W:
                kpos = (mx.arange(S) + (off + T - S))[None, :]
                qpos = (mx.arange(T) + off)[:, None]
                mask = (kpos <= qpos) & (kpos > qpos - W)
        elif T > 1:
            mask = "causal"
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(d), mask=mask
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, T, H * d)
        return self.o_proj(gate_out(o, g, sg2))


class DoryMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.I = args.intermediate_size
        self.gate_up_proj = nn.Linear(
            args.hidden_size, 2 * args.intermediate_size, bias=False
        )  # [gate | up]
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.suv_gate = mx.ones((args.intermediate_size,))
        self.suv_up = mx.ones((args.intermediate_size,))
        self._s = math.sqrt(args.hidden_size)
        self._scales = None

    def __call__(self, x):
        if self._scales is None:
            self._scales = (
                self.suv_gate.astype(mx.float32) * self._s,
                self.suv_up.astype(mx.float32) * self._s,
            )
            mx.eval(self._scales)
        sg, su = self._scales
        gu = self.gate_up_proj(x)
        return self.down_proj(mlp_act(gu[..., : self.I], gu[..., self.I :], sg, su))


class DoryLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.is_attn = layer_type != "mlp"
        self.layer_type = layer_type
        self._alpha_scale = args.ngpt_alpha_init / args.initializer_range
        self._alpha = None
        if self.is_attn:
            self.attn_alpha = mx.ones((args.hidden_size,))
            self.mixer = DoryAttention(args, layer_type == "sliding_attention")
        else:
            self.mlp_alpha = mx.ones((args.hidden_size,))
            self.mixer = DoryMLP(args)

    def __call__(self, h, cache=None, reset_to=None):
        if self._alpha is None:
            a = self.attn_alpha if self.is_attn else self.mlp_alpha
            self._alpha = mx.abs(a.astype(mx.float32) * self._alpha_scale)
            mx.eval(self._alpha)
        out = self.mixer(h, cache, reset_to) if self.is_attn else self.mixer(h)
        return slerp_norm(h, out, self._alpha)


class DoryBackbone(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DoryLayer(args, t) for t in args.layer_types]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = DoryBackbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logit_scale = mx.ones((args.vocab_size,))
        self._plan = None
        self._ls = None

    @property
    def layers(self):
        return self.backbone.layers

    def _cache_plan(self):
        if self._plan is None:
            a = self.args
            lo, hi = a.n_input_layers, a.n_input_layers + a.n_recurrent_layers
            L = a.n_recurrent_loops if a.recurrent_kv_cache_mode == "per_loop" else 1
            plan, n = {}, 0
            for i, t in enumerate(a.layer_types):
                if t == "mlp":
                    continue
                if lo <= i < hi:
                    for l in range(L):
                        plan[(i, l)] = n
                        n += 1
                else:
                    plan[(i, 0)] = n
                    n += 1
            self._plan = (plan, n)
        return self._plan

    def make_cache(self):
        plan, n = self._cache_plan()
        a = self.args
        caches = [None] * n
        use_sliding = a.recurrent_kv_cache_mode == "per_loop"
        for (i, _l), idx in plan.items():
            if use_sliding and a.layer_types[i] == "sliding_attention":
                caches[idx] = RingKVCache(a.sliding_window)
            else:
                caches[idx] = KVCache()
        return caches

    def __call__(self, inputs, cache=None, input_embeddings=None):
        a = self.args
        plan, n = self._cache_plan()
        if cache is None:
            cache = [None] * n
        h = (
            self.backbone.embeddings(inputs)
            if input_embeddings is None
            else input_embeddings
        )
        lo, hi = a.n_input_layers, a.n_input_layers + a.n_recurrent_layers
        layers = self.backbone.layers
        for i in range(lo):
            c = cache[plan[(i, 0)]] if layers[i].is_attn else None
            h = layers[i](h, c)
        input_h = h
        state = input_h
        per_loop = a.recurrent_kv_cache_mode == "per_loop"
        starts = {}
        for l in range(a.n_recurrent_loops):
            h = loop_inject(state, input_h)
            for i in range(lo, hi):
                if layers[i].is_attn:
                    c = cache[plan[(i, l if per_loop else 0)]]
                    reset = None
                    if not per_loop and c is not None:
                        if l == 0:
                            starts[i] = c.offset
                        else:
                            reset = starts[i]
                    h = layers[i](h, c, reset)
                else:
                    h = layers[i](h)
            state = justnorm(h)
        h = state
        for i in range(hi, a.num_hidden_layers):
            c = cache[plan[(i, 0)]] if layers[i].is_attn else None
            h = layers[i](h, c)
        if self._ls is None:
            self._ls = self.logit_scale.astype(mx.float32) / a.initializer_range
            mx.eval(self._ls)
        return (self.lm_head(h).astype(mx.float32) * self._ls).astype(h.dtype)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k == "lm_head.logit_scale":
                k = "logit_scale"
            out[k] = v
        for k in list(out.keys()):
            if k.endswith(".mixer.q_proj.weight"):
                p = k[: -len("q_proj.weight")]
                out[p + "qkvg_proj.weight"] = mx.concatenate(
                    [
                        out.pop(p + "q_proj.weight"),
                        out.pop(p + "k_proj.weight"),
                        out.pop(p + "v_proj.weight"),
                        out.pop(p + "g_proj.weight"),
                    ],
                    axis=0,
                )
            elif k.endswith(".mixer.gate_proj.weight"):
                p = k[: -len("gate_proj.weight")]
                out[p + "gate_up_proj.weight"] = mx.concatenate(
                    [out.pop(p + "gate_proj.weight"), out.pop(p + "up_proj.weight")],
                    axis=0,
                )
        return out
