# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .ssm import ssm_update
from .switch_layers import SwitchMLP


@dataclass()
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    mlp_bias: bool
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    hybrid_override_pattern: Optional[List[str]] = None
    layers_block_type: Optional[List[str]] = None
    head_dim: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_latent_size: Optional[int] = None
    n_group: Optional[int] = None
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    topk_group: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    norm_topk_prob: Optional[bool] = None
    routed_scaling_factor: Optional[float] = None
    time_step_limit: Optional[Tuple[float, float]] = None
    time_step_min: Optional[float] = None
    time_step_max: Optional[float] = None
    num_nextn_predict_layers: int = 0
    mtp_hybrid_override_pattern: Optional[str] = None
    mtp_layers_block_type: Optional[List[str]] = None

    # Map from layers_block_type names to single-char pattern codes
    _block_type_to_char = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}

    def __post_init__(self):
        if self.time_step_limit is None:
            self.time_step_limit = (0.0, float("inf"))

        # Normalize to hybrid_override_pattern (single-char list)
        if self.hybrid_override_pattern is None and self.layers_block_type is not None:
            self.hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.layers_block_type
            ]
        if self.hybrid_override_pattern is not None:
            self.num_hidden_layers = len(self.hybrid_override_pattern)

        # Normalize MTP pattern
        if self.mtp_hybrid_override_pattern is not None:
            if isinstance(self.mtp_hybrid_override_pattern, str):
                self._mtp_pattern = list(self.mtp_hybrid_override_pattern)
            else:
                self._mtp_pattern = list(self.mtp_hybrid_override_pattern)
        elif self.mtp_layers_block_type is not None:
            self._mtp_pattern = [
                self._block_type_to_char[t] for t in self.mtp_layers_block_type
            ]
        else:
            self._mtp_pattern = []


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, group_size: int):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)
        self.group_size = group_size

    def __call__(self, x: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            x = swiglu(gate, x)
        x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.flatten(-2)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        group_size = self.intermediate_size // self.n_groups
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=args.layer_norm_epsilon,
            group_size=group_size,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None

        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D.astype(hidden_states.dtype),
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
        )
        if cache:
            cache[1] = state

        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:

        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = (
            args.head_dim
            if args.head_dim is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size

        self.up_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x):
        return self.down_proj(nn.relu2(self.up_proj(x)))


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

    orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
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

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class NemotronHMoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_latent_size = config.moe_latent_size

        # When latent projection is used, experts operate on the latent dim
        expert_input_dim = (
            config.moe_latent_size
            if config.moe_latent_size is not None
            else config.hidden_size
        )
        self.switch_mlp = SwitchMLP(
            expert_input_dim,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=nn.ReLU2(),
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_shared_expert_intermediate_size
            self.shared_experts = NemotronHMLP(
                config, intermediate_size=intermediate_size
            )

        # Latent projection layers for dimensionality reduction before/after experts
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )

    def __call__(self, x):
        residuals = x
        inds, scores = self.gate(x)

        if self.moe_latent_size is not None:
            x = self.fc1_latent_proj(x)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.moe_latent_size is not None:
            y = self.fc2_latent_proj(y)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(residuals)

        return y


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, block_type: str):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.block_type = block_type

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif self.block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif self.block_type == "-":
            self.mixer = NemotronHMLP(args)
        elif self.block_type == "E":
            self.mixer = NemotronHMoE(args)

    def __call__(
        self,
        x,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.norm(x)
        if self.block_type == "M" or self.block_type == "*":
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)

        return x + hidden_states


class NemotronHMTPBlock(nn.Module):
    """A single block in the MTP head. Follows the same pattern as
    NemotronHBlock but only supports attention ('*') and MoE ('E') types,
    matching the ``mtp_hybrid_override_pattern``."""

    def __init__(self, args: ModelArgs, block_type: str):
        super().__init__()
        self.block_type = block_type
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        if block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif block_type == "E":
            self.mixer = NemotronHMoE(args)
        elif block_type == "-":
            self.mixer = NemotronHMLP(args)
        else:
            raise ValueError(f"Unsupported MTP block type: {block_type}")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.norm(x)
        if self.block_type == "*":
            h = self.mixer(h, mask=mask, cache=cache)
        else:
            h = self.mixer(h)
        return x + h


class NemotronHMTPModule(nn.Module):
    """Multi-Token Prediction head for Nemotron-H.

    Predicts token t+2 from the backbone's pre-norm hidden state h_t and the
    sampled token t+1, using a shared ``lm_head`` with the backbone.

    Architecture (for ``mtp_hybrid_override_pattern = "*E"``):
      1. Embed next_token via shared embedding
      2. Dual-norm fusion: ``eh_proj(cat(enorm(embed), hnorm(hidden)))``
      3. Attention block (``*``) with its own KVCache
      4. MoE block (``E``) — same structure as backbone MoE
      5. Final layernorm

    Weight mapping from HF checkpoints:
      ``mtp.layers.0.hnorm``   → ``hnorm``
      ``mtp.layers.0.enorm``   → ``enorm``
      ``mtp.layers.0.eh_proj`` → ``eh_proj``
      ``mtp.layers.0.norm``    → ``layers.0.norm``
      ``mtp.layers.0.mixer.*`` → ``layers.0.mixer.*``
      ``mtp.layers.1.norm``    → ``layers.1.norm``
      ``mtp.layers.1.mixer.*`` → ``layers.1.mixer.*``
      ``mtp.layers.1.final_layernorm`` → ``final_layernorm``
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [NemotronHMTPBlock(args, bt) for bt in args._mtp_pattern]
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Embedding,
        cache: Optional[Any] = None,
    ) -> mx.array:
        embeds = embed_tokens(next_token_ids)
        e = self.enorm(embeds)
        h = self.hnorm(hidden_states)
        fused = self.eh_proj(mx.concatenate([e, h], axis=-1))

        if cache is None:
            cache = [None] * len(self.layers)

        # Build attention mask from the first attention layer's cache
        attn_cache_idx = 0
        for i, layer in enumerate(self.layers):
            if layer.block_type == "*":
                attn_cache_idx = i
                break
        mask = create_attention_mask(fused, cache[attn_cache_idx])

        cache_idx = 0
        for layer in self.layers:
            if layer.block_type == "*":
                fused = layer(fused, mask=mask, cache=cache[cache_idx])
                cache_idx += 1
            else:
                fused = layer(fused)

        return self.final_layernorm(fused)


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args, block_type)
            for block_type in args.hybrid_override_pattern
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in args.hybrid_override_pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(
        self,
        inputs,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return hidden_states


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type
        if args.num_nextn_predict_layers > 0 and len(args._mtp_pattern) > 0:
            self.mtp = NemotronHMTPModule(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        hidden = self.backbone(inputs, cache=cache)
        out = self.lm_head(self.backbone.norm_f(hidden))
        if return_hidden:
            return out, hidden
        return out

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: Any,
    ) -> mx.array:
        """Run the MTP head and apply the shared lm_head.

        Args:
            hidden_states: (B, 1, H) — backbone pre-norm hidden at last position.
            next_token_ids: (B, 1) — sampled main token.
            mtp_cache: list of KVCache entries for MTP attention layers.

        Returns:
            logits: (B, 1, vocab_size)
        """
        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.backbone.embeddings,
            mtp_cache,
        )
        return self.lm_head(mtp_out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif l.block_type == "*":
                caches.append(KVCache())
        return caches

    def make_mtp_cache(self):
        """Return a fresh list of KVCache entries for MTP attention layers."""
        if hasattr(self, "mtp"):
            return [KVCache() for layer in self.mtp.layers if layer.block_type == "*"]
        return []

    def sanitize(self, weights):
        has_mtp = self.args.num_nextn_predict_layers > 0
        if not has_mtp:
            weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}

        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        # Stack backbone experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"backbone.layers.{l}.mixer"
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                if f"{prefix}.experts.0.{m}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        if has_mtp:
            # Remap MTP weights from HF naming to our module structure.
            #
            # HF layout (num_nextn_predict_layers=1, pattern "*E"):
            #   mtp.layers.0.hnorm        → fusion norms
            #   mtp.layers.0.enorm
            #   mtp.layers.0.eh_proj       → fusion projection
            #   mtp.layers.0.norm          → attention block norm
            #   mtp.layers.0.mixer.*       → attention block mixer
            #   mtp.layers.1.norm          → MoE block norm
            #   mtp.layers.1.mixer.*       → MoE block mixer
            #   mtp.layers.1.final_layernorm → final output norm
            #
            # Our layout:
            #   mtp.hnorm, mtp.enorm, mtp.eh_proj
            #   mtp.layers.0.norm, mtp.layers.0.mixer.*  (attention)
            #   mtp.layers.1.norm, mtp.layers.1.mixer.*  (MoE)
            #   mtp.final_layernorm

            remap = {}
            mtp_keys = [k for k in weights if k.startswith("mtp.")]
            for k in mtp_keys:
                v = weights.pop(k)
                rest = k[len("mtp.") :]

                # Fusion components live on HF layer 0
                if rest.startswith("layers.0.hnorm."):
                    new_k = "mtp." + rest.replace("layers.0.hnorm.", "hnorm.")
                elif rest.startswith("layers.0.enorm."):
                    new_k = "mtp." + rest.replace("layers.0.enorm.", "enorm.")
                elif rest.startswith("layers.0.eh_proj."):
                    new_k = "mtp." + rest.replace("layers.0.eh_proj.", "eh_proj.")
                # Attention block: HF layer 0 norm/mixer → our layers.0
                elif rest.startswith("layers.0.norm."):
                    new_k = "mtp.layers.0.norm." + rest[len("layers.0.norm.") :]
                elif rest.startswith("layers.0.mixer."):
                    new_k = "mtp.layers.0.mixer." + rest[len("layers.0.mixer.") :]
                # MoE block: HF layer 1 → our layers.1
                elif rest.startswith("layers.1.norm."):
                    new_k = "mtp.layers.1.norm." + rest[len("layers.1.norm.") :]
                elif rest.startswith("layers.1.final_layernorm."):
                    new_k = (
                        "mtp.final_layernorm."
                        + rest[len("layers.1.final_layernorm.") :]
                    )
                elif rest.startswith("layers.1.mixer."):
                    new_k = "mtp.layers.1.mixer." + rest[len("layers.1.mixer.") :]
                else:
                    new_k = "mtp." + rest

                remap[new_k] = v

            # Stack MTP MoE experts (same pattern as backbone)
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                expert_key = f"mtp.layers.1.mixer.experts.0.{m}.weight"
                if expert_key in remap:
                    to_join = [
                        remap.pop(f"mtp.layers.1.mixer.experts.{e}.{m}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    remap[f"mtp.layers.1.mixer.switch_mlp.{n}.weight"] = mx.stack(
                        to_join
                    )

            weights.update(remap)

        return weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k and "A_log" not in k

        return predicate
