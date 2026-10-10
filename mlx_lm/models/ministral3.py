# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    QuantizedAllToShardedLinear,
    QuantizedShardedToAllLinear,
    shard_linear,
)
from mlx.utils import tree_unflatten

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    gather_last_axis,
    scaled_dot_product_attention,
)
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


def _get_llama_4_attn_scale(size, offset, beta: float, max_position_embeddings: int):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]

    scaling = 1 + beta * mx.log(
        1 + mx.floor((mx.arange(size) + offset) / max_position_embeddings)
    )
    if scaling.ndim == 2:
        return scaling[:, None, :, None]
    else:
        return scaling[:, None]


class Fp8Linear(nn.QuantizedLinear):
    """Linear on e4m3 weights, run as mxfp8 with unit group scales.

    The kernel has no slot for the per-tensor scale, so it scales the output.
    Fused layers have one scale per output row.
    """

    def __init__(self, input_dims: int, output_dims: int, per_row: bool = False):
        super().__init__(input_dims, output_dims, False, 32, 8, "mxfp8")
        self.output_scale = mx.ones((output_dims,)) if per_row else mx.array(1.0)
        self.freeze()

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class Fp8AllToShardedLinear(QuantizedAllToShardedLinear):
    """Fp8Linear with the outputs sharded across the group."""

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class Fp8ShardedToAllLinear(QuantizedShardedToAllLinear):
    """Fp8Linear with the inputs sharded across the group."""

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


def _shard_linear(layer, sharding, group, segments=1):
    if not isinstance(layer, Fp8Linear):
        return shard_linear(layer, sharding, segments=segments, group=group)
    if sharding == "all-to-sharded":
        cls = Fp8AllToShardedLinear
    else:
        cls = Fp8ShardedToAllLinear
    scale = layer.pop("output_scale")
    sharded = cls.from_quantized_linear(layer, segments=segments, group=group)
    # A per-row scale follows the output rows. A scalar scale applies to each
    # shard and to each partial sum, so every rank keeps all of it.
    if scale.ndim == 1:
        N, r = group.size(), group.rank()
        parts = mx.split(scale, segments)
        scale = mx.concatenate([mx.split(p, N)[r] for p in parts])
    sharded.output_scale = scale
    sharded.freeze()
    return sharded


def _fuse(weights, parts, fused):
    """Join the output rows of layers that read the same input.

    Returns False and changes nothing when the layers are stored differently,
    for example quantized with different bits.
    """
    keys = [{k[len(p) + 1 :] for k in weights if k.startswith(p + ".")} for p in parts]
    if "weight" not in keys[0] or any(k != keys[0] for k in keys):
        return False
    for k in keys[0]:
        vs = [weights[f"{p}.{k}"] for p in parts]
        if k == "output_scale" and vs[0].ndim > 0:
            return False
        if k not in ("output_scale", "bias") and any(
            v.dtype != vs[0].dtype or v.shape[-1] != vs[0].shape[-1] for v in vs
        ):
            return False

    rows = [weights[f"{p}.weight"].shape[-2] for p in parts]
    for k in keys[0]:
        vs = [weights.pop(f"{p}.{k}") for p in parts]
        if k == "output_scale":
            # Each fp8 layer has one scale. Fused, there is one per row.
            vs = [mx.full((n,), s, dtype=s.dtype) for n, s in zip(rows, vs)]
            weights[f"{fused}.{k}"] = mx.concatenate(vs)
        else:
            axis = -1 if k == "bias" else -2
            weights[f"{fused}.{k}"] = mx.concatenate(vs, axis=axis)
    return True


def _fp8_as_mxfp8(weights):
    """Read the e4m3 weights of fp8 layers as mxfp8 with unit (2^0) group scales."""
    for k in [k for k in weights if k.endswith(".output_scale")]:
        p = k.removesuffix(".output_scale")
        w = weights[f"{p}.weight"]
        if w.dtype == mx.uint8:
            weights[f"{p}.weight"] = w.view(mx.uint32)
            weights[f"{p}.scales"] = mx.full(
                (*w.shape[:-1], w.shape[-1] // 32), 127, mx.uint8
            )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        self.scaling_beta = args.rope_parameters["llama_4_scaling_beta"]
        self.original_max_position_embeddings = args.rope_parameters[
            "original_max_position_embeddings"
        ]

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_parameters["rope_theta"],
            False,
            args.rope_parameters,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # sanitize fuses the three projections unless they are stored differently.
        if "qkv_proj" in self:
            q, kv = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            queries, keys, values = mx.split(self.qkv_proj(x), [q, q + kv], axis=-1)
        else:
            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        if attn_scale is None:
            attn_scale = _get_llama_4_attn_scale(
                L,
                offset,
                self.scaling_beta,
                self.original_max_position_embeddings,
            ).astype(x.dtype)
        queries = queries * attn_scale
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class FusedGLU(nn.Module):
    """Gated MLP with gate_proj and up_proj fused into one projection."""

    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dims, 2 * hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, input_dims, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        attn_scale: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, l in enumerate(self.layers):
            if l.use_sliding:
                self.swa_idx = e
                break

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self.fa_idx = None
        self.swa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.swa_idx is None and l.use_sliding:
                self.swa_idx = e
            elif self.fa_idx is None and not l.use_sliding:
                self.fa_idx = e
            if self.fa_idx is not None and self.swa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
            offset = 0
        else:
            offset = cache[0].offset

        swa_mask = fa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        attn_scale = _get_llama_4_attn_scale(
            inputs.shape[1],
            offset,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            mask = swa_mask if l.use_sliding else fa_mask
            h = l(h, attn_scale, mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
            if self.sharding_group is not None:
                out = gather_last_axis(out, self.sharding_group)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # fp8 layers keep the e4m3 bytes, and the scale moves to the layer.
        # Static activation scales have no consumer here.
        weights = {
            k.replace(".weight_scale_inv", ".output_scale"): v
            for k, v in weights.items()
            if "activation_scale" not in k
        }

        def fused(parts, name):
            if f"{name}.weight" in weights:
                return True
            # Only fp8 layers are fused. AWQ and the mixed-bit recipes look up
            # q_proj, v_proj and gate_proj by name.
            fp8 = f"{parts[0]}.output_scale" in weights
            return fp8 and _fuse(weights, parts, name)

        # Projections that read the same input run as one matmul.
        for l, layer in enumerate(self.model.layers):
            attn, mlp = f"model.layers.{l}.self_attn", f"model.layers.{l}.mlp"
            a = layer.self_attn
            if fused([f"{attn}.{n}_proj" for n in "qkv"], f"{attn}.qkv_proj"):
                rows = (a.n_heads + 2 * a.n_kv_heads) * a.head_dim
                a.qkv_proj = nn.Linear(self.args.hidden_size, rows, bias=False)
                del a.q_proj, a.k_proj, a.v_proj
            if fused([f"{mlp}.gate_proj", f"{mlp}.up_proj"], f"{mlp}.gate_up_proj"):
                out_dims, in_dims = layer.mlp.gate_proj.weight.shape
                layer.mlp = FusedGLU(in_dims, out_dims)

        _fp8_as_mxfp8(weights)
        fp8_modules = []
        for p, m in self.named_modules():
            if isinstance(m, nn.Linear) and f"{p}.output_scale" in weights:
                out_dims, in_dims = m.weight.shape
                per_row = weights[f"{p}.output_scale"].ndim == 1
                fp8_modules.append((p, Fp8Linear(in_dims, out_dims, per_row)))
        if fp8_modules:
            self.update_modules(tree_unflatten(fp8_modules))

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        if not self.args.tie_word_embeddings:
            vocab = self.args.vocab_size
            assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
            self.lm_head = _shard_linear(self.lm_head, "all-to-sharded", group)
            self.sharding_group = group

        for layer in self.model.layers:
            # Shard the self attention
            attn = layer.self_attn
            if "qkv_proj" in attn:
                # Each rank takes its heads of q, k and v.
                q = attn.n_heads * attn.head_dim
                kv = attn.n_kv_heads * attn.head_dim
                attn.qkv_proj = _shard_linear(
                    attn.qkv_proj, "all-to-sharded", group, segments=[q, q + kv]
                )
            else:
                attn.q_proj = _shard_linear(attn.q_proj, "all-to-sharded", group)
                attn.k_proj = _shard_linear(attn.k_proj, "all-to-sharded", group)
                attn.v_proj = _shard_linear(attn.v_proj, "all-to-sharded", group)
            attn.o_proj = _shard_linear(attn.o_proj, "sharded-to-all", group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            # Shard the MLP
            mlp = layer.mlp
            if isinstance(mlp, FusedGLU):
                mlp.gate_up_proj = _shard_linear(
                    mlp.gate_up_proj, "all-to-sharded", group, segments=2
                )
            else:
                mlp.gate_proj = _shard_linear(mlp.gate_proj, "all-to-sharded", group)
                mlp.up_proj = _shard_linear(mlp.up_proj, "all-to-sharded", group)
            mlp.down_proj = _shard_linear(mlp.down_proj, "sharded-to-all", group)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
