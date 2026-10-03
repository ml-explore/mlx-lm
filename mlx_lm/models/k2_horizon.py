# Copyright © 2026 Apple Inc.

"""K2-Horizon for MLX: MoE with Mixture-of-Values (MoVA) attention.

This file is an MLX port of the official PyTorch implementation,
``modeling_k2_horizon.py`` in IFM/K2-Horizon-MoVA-36B-A4B (revision e5c131d).
The math is the same. The implementation is different in these ways:

1. Experts are computed with ``gather_mm`` (``SwitchGLU`` / ``SwitchLinear``).
   The reference loops over experts in Python, finds each expert's tokens with
   ``one_hot`` / ``torch.where`` and accumulates with ``index_add_``. That style
   needs in-place updates and data-dependent control flow, which are slow or not
   available in MLX's lazy graph, so all selected experts run in one kernel call.
2. Per-expert weights are stacked into one tensor per projection when the
   checkpoint loads (see ``Model.sanitize``).
3. ``K2HorizonAttention`` and ``K2HorizonMoVAAttention`` are merged into one
   ``Attention`` class. Only the source of the values differs.
4. Attention, RoPE and RMSNorm use MLX fused kernels (``mx.fast.*``) instead of
   the eager PyTorch ops.
5. In bf16, expert outputs are weighted and summed in fp32. The reference casts
   the routing weights to bf16 and accumulates in bf16. The result is slightly
   more precise, so bf16 outputs differ from bf16 PyTorch by rounding only.
6. KV cache and masks use mlx-lm's cache objects, not the HF ``Cache`` classes.

Checked against the reference in fp32 on the real weights: every layer matches
to a relative error of 1e-6 or better, and greedy decoding with the KV cache
gives the same tokens.

Not supported (the reference supports them, this checkpoint does not use them):
partial RoPE (``rope_head_dim != head_dim``), sliding-window attention, and the
training-only parts (dropout, router logits output, load-balancing loss).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear, _gather_sort, _scatter_unsort


@dataclass
class ModelArgs(BaseModelArgs):
    """Model config. Same fields and defaults as ``K2HorizonConfig``.

    Fields that only matter for training (``router_aux_loss_coef``,
    ``attention_dropout``, ``output_router_logits``) are ignored.
    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[List[int]] = None
    norm_topk_prob: bool = False
    num_shared_experts: int = 0
    moe_gate_bias: bool = False
    router_score_func: str = "softmax"
    router_scaling_factor: Optional[float] = 1.0
    mova_num_experts: int = 0
    mova_num_experts_per_tok: int = 0
    attention_bias: bool = False
    attention_gate_func: Optional[str] = None
    query_key_norm: bool = True
    layernorm_num_groups: int = 1
    rope_head_dim: Optional[int] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    use_sliding_window: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Same defaults as K2HorizonConfig.__post_init__.
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        if self.router_scaling_factor is None:
            self.router_scaling_factor = 1.0
        # Newer configs keep rope_theta inside rope_parameters.
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        # The reference supports these two features. They are not ported, so fail
        # loudly instead of computing wrong outputs.
        if self.rope_head_dim is not None and self.rope_head_dim != self.head_dim:
            raise NotImplementedError("Partial RoPE is not supported for k2_horizon.")
        if self.use_sliding_window:
            raise NotImplementedError("Sliding window is not supported for k2_horizon.")
        # YaRN RoPE (used by K2-Horizon-0.9B) goes through mlx-lm's YarnRoPE.
        # Check that the config does not use options YarnRoPE handles differently.
        rope = self.rope_parameters or {}
        if rope.get("rope_type", rope.get("type")) == "yarn":
            _check_yarn(rope)


def _check_yarn(rope):
    """Reject YaRN configs where mlx-lm's YarnRoPE differs from the reference.

    The reference uses transformers' ``_compute_yarn_parameters``. mlx-lm's
    ``YarnRoPE`` computes the same frequencies, with two differences:

    - It always truncates the correction range (floor/ceil). The reference does
      that only when ``truncate`` is True, which is the default.
    - It always derives the attention scale from ``factor``, ``mscale`` and
      ``mscale_all_dim``. The reference uses ``attention_factor`` directly when
      the config gives one, and treats ``mscale`` differently when
      ``mscale_all_dim`` is missing.

    K2-Horizon-0.9B sets truncate=True and attention_factor=1.27726, which is
    exactly 0.1 * ln(16) + 1, so both agree. Other values raise here instead of
    giving wrong outputs.
    """
    if not rope.get("truncate", True):
        raise NotImplementedError("YaRN with truncate=False is not supported.")
    if rope.get("factor") is None:
        # The reference would derive factor from max_position_embeddings.
        raise NotImplementedError("YaRN without a factor is not supported.")

    def get_mscale(scale, m=1):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    factor = rope["factor"]
    mscale, mscale_all_dim = rope.get("mscale"), rope.get("mscale_all_dim")
    # The scale YarnRoPE applies (its defaults are mscale=1, mscale_all_dim=0).
    ours = get_mscale(factor, mscale or 1) / get_mscale(factor, mscale_all_dim or 0)
    # The scale the reference applies.
    if rope.get("attention_factor") is not None:
        ref = rope["attention_factor"]
    elif mscale and mscale_all_dim:
        ref = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
    else:
        ref = get_mscale(factor)
    if abs(ours - ref) > 1e-6:
        raise NotImplementedError(
            f"YaRN attention factor {ref} is not supported (MLX would use {ours})."
        )


class GroupedRMSNorm(nn.Module):
    """RMSNorm applied independently to ``groups`` equal slices of the input.

    Reference: ``K2HorizonRMSNorm``. Same math: reshape to (..., groups, dims //
    groups), normalize each group in fp32, reshape back, multiply by the
    full-width weight, cast back to the input dtype. The per-group
    normalization uses the fused ``mx.fast.rms_norm`` kernel.
    """

    def __init__(self, dims: int, groups: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x):
        if self.groups == 1:
            # Fast path: the fused kernel also applies the weight. In bf16 this
            # can round slightly differently from the reference. This checkpoint
            # uses groups > 1 for all its norms, so it does not take this path.
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        y = x.astype(mx.float32).reshape(*shape[:-1], self.groups, -1)
        y = mx.fast.rms_norm(y, None, self.eps).reshape(shape)
        return (self.weight * y).astype(x.dtype)


class Router(nn.Module):
    """Top-k router shared by the MoE block and MoVA attention.

    Reference: ``calc_router_weights`` (used by MoVA) and the routing code at the
    start of ``K2HorizonSparseMoeBlock.forward``. Both do:

    1. logits = x @ W.T, without the bias (the reference calls ``F.linear`` with
       the weight only)
    2. scores = sigmoid(logits) or softmax(logits), in fp32
    3. select the top-k experts using scores + bias
    4. weights = the *unbiased* scores of the selected experts
    5. optionally renormalize the weights to sum to 1, then multiply by
       ``router_scaling_factor``

    So the bias only changes *which* experts are selected, never the weights.
    Getting this wrong changes the output a lot, and the parity tests check it.

    Differences from the reference:
    - ``torch.topk`` is replaced by ``mx.argpartition``, which does not sort the
      selected experts. The order does not matter because the outputs are
      summed. Results differ only when two scores are exactly equal.
    - The weights stay in fp32. The MoE reference casts them to the hidden dtype
      (bf16) before use.
    """

    def __init__(
        self,
        dims: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        scaling_factor: float,
        renormalize: bool,
        bias: bool,
    ):
        super().__init__()
        # Parameter names match the checkpoint: ``mlp.gate.{weight,bias}`` and
        # ``self_attn.v_router.{weight,bias}``. This is a custom module, not
        # nn.Linear, so quantization leaves the routers in full precision.
        self.weight = mx.zeros((num_experts, dims))
        if bias:
            self.bias = mx.zeros((num_experts,))
        self.top_k = top_k
        self.score_func = score_func
        self.scaling_factor = scaling_factor
        self.renormalize = renormalize

    def __call__(self, x):
        # Same as the reference: matmul in the model dtype, scores in fp32.
        logits = (x @ self.weight.T).astype(mx.float32)
        if self.score_func == "sigmoid":
            scores = mx.sigmoid(logits)
        elif self.score_func == "softmax":
            scores = mx.softmax(logits, axis=-1, precise=True)
        else:
            raise ValueError(f"Unsupported router score function: {self.score_func}")

        # The bias is used for selection only.
        selection = scores
        if "bias" in self:
            selection = selection + self.bias.astype(mx.float32)

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k])
        # Gather the unbiased scores, as torch.gather(routing_scores, ...) does.
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.renormalize:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return inds, weights * self.scaling_factor


class SwitchValueExperts(nn.Module):
    """MoVA values: sum_k w_k * silu(x @ V_k) over the selected value experts.

    Reference: ``combine_routed_experts(..., activation=F.silu)`` called from
    ``K2HorizonMoVAAttention.forward``, with ``v_experts`` as an
    ``nn.ModuleList`` of 64 bias-free ``nn.Linear`` layers.

    Differences from the reference:
    - The reference loops over the experts that were hit, runs each one on its
      tokens, and accumulates with ``index_add_``. Here one ``gather_mm`` call
      runs all selected experts. The 64 expert weights are stacked into one
      [64, kv_heads * head_dim, hidden] tensor by ``Model.sanitize``.
    - The weighted sum is done in fp32 and cast back once. The reference
      multiplies and accumulates in bf16.

    Same as the reference: SiLU is applied to each expert output *before* the
    weighted sum, not after.
    """

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__()
        self.proj = SwitchLinear(input_dims, output_dims, num_experts, bias=False)

    def __call__(self, x, indices, weights):
        x = mx.expand_dims(x, (-2, -3))
        # Same as SwitchGLU: when there are many (token, expert) pairs, as in
        # prompt processing, sort them by expert so that gather_mm reads each
        # expert's weights in order. For one-token decode steps this is skipped.
        do_sort = indices.size >= 64
        idx = indices
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x = nn.silu(self.proj(x, idx, sorted_indices=do_sort))
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        x = x.squeeze(-2)
        return (x * weights[..., None]).sum(axis=-2).astype(x.dtype)


class Attention(nn.Module):
    """GQA attention with an optional MoVA value path and output gate.

    Reference: ``K2HorizonAttention`` (dense layers 0-2) and
    ``K2HorizonMoVAAttention`` (sparse layers). The two reference classes are
    the same except for the values, so here they are one class:
    - ``mova=False``: values = v_proj(x)
    - ``mova=True``: values come from the routed value experts (``v_router`` +
      ``v_experts``). There is no ``v_proj`` in these layers.

    The values are computed per token from the layer input, so they go into the
    KV cache like normal values. MoVA needs no special cache.

    Differences from the reference:
    - RoPE: ``mx.fast.rope`` with ``traditional=False``, which is the same
      rotate-half layout as the reference ``apply_rotary_pos_emb``. The
      reference makes cos/sin tables and casts them to the model dtype. The
      fused kernel computes them itself, which can round differently in bf16.
      Positions come from the cache offset, not a ``position_ids`` argument.
    - Attention: the fused ``scaled_dot_product_attention`` kernel with native
      GQA. The reference eager path repeats the KV heads (``repeat_kv``) and
      computes softmax in fp32.
    - Cache and mask: mlx-lm ``KVCache`` and causal mask, not HF ``Cache`` and
      ``create_causal_mask``.
    """

    def __init__(self, args: ModelArgs, mova: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5
        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        self.mova = mova
        if mova:
            self.v_router = Router(
                dim,
                args.mova_num_experts,
                args.mova_num_experts_per_tok,
                args.router_score_func,
                args.router_scaling_factor,
                # calc_router_weights ignores norm_topk_prob. It renormalizes
                # only when top_k > 1. The MoE block is different (see below).
                renormalize=args.mova_num_experts_per_tok > 1,
                bias=args.moe_gate_bias,
            )
            self.v_experts = SwitchValueExperts(
                dim, n_kv_heads * head_dim, args.mova_num_experts
            )
        else:
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)

        self.gate_func = args.attention_gate_func
        if self.gate_func is not None:
            if self.gate_func not in ("silu", "softplus"):
                raise ValueError(f"Unsupported attention gate: {self.gate_func}")
            self.gate_proj = nn.Linear(dim, n_heads * head_dim, bias=False)

        # Reference: q_norm / k_norm are K2HorizonRMSNorm with one group per
        # head, applied to the full projection before the reshape into heads.
        # This checkpoint has query_key_norm=False.
        self.query_key_norm = args.query_key_norm
        if self.query_key_norm:
            self.q_norm = GroupedRMSNorm(n_heads * head_dim, n_heads, args.rms_norm_eps)
            self.k_norm = GroupedRMSNorm(
                n_kv_heads * head_dim, n_kv_heads, args.rms_norm_eps
            )

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys = self.q_proj(x), self.k_proj(x)
        if self.query_key_norm:
            queries, keys = self.q_norm(queries), self.k_norm(keys)
        # MoVA: route each token to its value experts and mix their outputs.
        if self.mova:
            values = self.v_experts(x, *self.v_router(x))
        else:
            values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3)

        # Per-head output gate, applied to the attention output before o_proj,
        # same as the reference.
        if self.gate_func is not None:
            gate = self.gate_proj(x).reshape(B, L, self.n_heads, -1)
            if self.gate_func == "silu":
                gate = nn.silu(gate)
            else:
                # Reference: F.softplus(gate, beta=log(2)) = log2(1 + 2^x).
                # Computed in fp32 with logaddexp, which is stable for large x.
                # PyTorch switches to the identity when beta * x > 20; the
                # difference is about e^-20.
                g = gate.astype(mx.float32) * math.log(2)
                gate = (mx.logaddexp(g, 0.0) / math.log(2)).astype(output.dtype)
            output = output * gate

        return self.o_proj(output.reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP. Same as ``K2HorizonMLP``.

    Used for the dense layers (``mlp_only_layers``) and for the shared expert.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    """Routed SwiGLU experts plus optional shared experts.

    Reference: ``K2HorizonSparseMoeBlock``.

    Differences from the reference:
    - The routed experts are one ``SwitchGLU`` (gather_mm) instead of an
      ``nn.ModuleList`` of ``K2HorizonMLP`` run in a Python loop.
    - The weighted sum of expert outputs is done in fp32. The reference casts
      the routing weights to bf16 and accumulates with ``index_add_`` in bf16.
    - Returns only the hidden states. The reference also returns the router
      logits, which are only used for the training loss.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.gate = Router(
            dim,
            args.num_experts,
            args.num_experts_per_tok,
            args.router_score_func,
            args.router_scaling_factor,
            # The MoE block renormalizes whenever norm_topk_prob is set, also
            # for top-1 (the weight becomes 1.0). MoVA does not (see Attention).
            renormalize=args.norm_topk_prob,
            bias=args.moe_gate_bias,
        )
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        # Same as the reference: the shared experts are one wider MLP, and its
        # output is added to the routed output without a gate.
        if args.num_shared_experts > 0:
            self.shared_experts = MLP(
                dim, args.moe_intermediate_size * args.num_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if "shared_experts" in self:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer. Same structure as ``K2HorizonDecoderLayer``."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        # Same rule as the reference. A sparse layer uses both the MoE block and
        # MoVA attention; a dense layer uses the dense MLP and v_proj attention.
        # This checkpoint: layers 0-2 are dense, layers 3-47 are sparse.
        is_sparse = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.self_attn = Attention(args, mova=is_sparse and args.mova_num_experts > 0)
        if is_sparse:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        self.input_layernorm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )
        self.post_attention_layernorm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class K2HorizonModel(nn.Module):
    """Embeddings, decoder layers and final norm. Reference: ``K2HorizonModel``.

    Differences from the reference:
    - The reference computes the RoPE cos/sin once per forward and passes them
      to every layer. Here each attention layer applies RoPE itself, using the
      cache offset as the start position.
    - The mask comes from ``create_attention_mask`` (mlx-lm), not HF
      ``create_causal_mask``. Padding in batches is handled by mlx-lm's batched
      generation, not by an ``attention_mask`` argument.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GroupedRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """Causal LM head. Reference: ``K2HorizonForCausalLM``.

    Returns logits only. The reference can also compute the LM loss and the
    load-balancing loss, which are only needed for training.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        """Convert the PyTorch checkpoint names and layout to this module.

        The reference stores every expert as its own nn.Linear. gather_mm needs
        one stacked tensor per projection, so:

        - ``model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight``
          (one per expert) -> ``model.layers.{l}.mlp.switch_mlp.{proj}.weight``
          with shape [num_experts, out, in]
        - ``model.layers.{l}.self_attn.v_experts.{e}.weight``
          -> ``model.layers.{l}.self_attn.v_experts.proj.weight``
          with shape [mova_num_experts, out, in]

        All other names match the reference unchanged. Already converted
        weights pass through, so the function is safe to run twice.
        """
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # RoPE frequencies are computed at run time, not loaded.
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            if f"{prefix}.mlp.experts.0.up_proj.weight" in weights:
                for n in ["up_proj", "down_proj", "gate_proj"]:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
            if f"{prefix}.self_attn.v_experts.0.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.self_attn.v_experts.{e}.weight")
                    for e in range(self.args.mova_num_experts)
                ]
                weights[f"{prefix}.self_attn.v_experts.proj.weight"] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers
