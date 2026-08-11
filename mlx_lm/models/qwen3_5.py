# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten, tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from .cache import ArraysCache, KVCache, NativeMTPRequestCache
from .gated_delta import gated_delta_update
from .pipeline import PipelineMixin
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

    # Native Qwen multi-token prediction layers.  A positive value only
    # declares the checkpoint format; ``supports_mtp`` stays false until
    # sanitization has validated that all of the head weights are present.
    mtp_num_hidden_layers: int = 0

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
        if self.mtp_num_hidden_layers < 0:
            raise ValueError("mtp_num_hidden_layers must be non-negative")
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


@dataclass(frozen=True)
class MTPCapability:
    """Immutable snapshot of whether this loaded model can run native MTP.

    This is deliberately model-local rather than a configuration flag.  A
    checkpoint which merely advertises MTP in config is not eligible until its
    sanitized weights have been checked, and PipelineMixin execution is never
    eligible because the head is not distributed with the backbone.
    """

    supported: bool
    reason: str
    num_layers: int


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
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
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
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
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

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
            cache[1] = state
            cache.advance(S)

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
            self.linear_attn = GatedDeltaNet(args)
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


class MTPDecoderLayer(nn.Module):
    """The full-attention transformer block used by a Qwen native MTP head.

    Qwen's backbone interleaves GatedDeltaNet and attention layers.  Its MTP
    head does not: it is a small attention-only decoder and consequently owns
    a separate, ordinary KV cache.
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = (
            SparseMoeBlock(args)
            if args.num_experts > 0
            else MLP(args.hidden_size, args.intermediate_size)
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MTPModule(nn.Module):
    """Qwen native next-depth prediction head.

    The final output head is intentionally shared with the backbone.  This
    module therefore produces hidden states only; TextModel applies the shared
    token projection after its final RMS norm.
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Embedding,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("MTP cache length does not match the number of MTP layers")

        embeddings = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        states = self.fc(mx.concatenate([embeddings, hidden_states], axis=-1))
        mask = create_attention_mask(states, cache[0]) if cache else None
        for layer, layer_cache in zip(self.layers, cache):
            states = layer(states, mask, layer_cache)
        return self.norm(states)


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_pre_norm: bool = False,
    ) -> mx.array:
        if return_pre_norm and self.pipeline_size != 1:
            raise RuntimeError(
                "Native Qwen MTP is not supported with pipeline parallelism"
            )
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        if return_pre_norm:
            return hidden_states
        return self.norm(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.mtp_num_hidden_layers:
            self.mtp = MTPModule(args)
        # This is only changed by sanitize after the complete MTP subtree has
        # been validated.  A config declaration alone must never enable MTP.
        self._mtp_weight_keys_validated = False
        self._mtp_weights_loaded = False
        self._mtp_pending_load_handshake = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ) -> mx.array:
        if return_hidden and not self.supports_mtp:
            raise RuntimeError(self.mtp_capability.reason)
        if return_hidden:
            hidden = self.model(
                inputs,
                cache,
                input_embeddings=input_embeddings,
                return_pre_norm=True,
            )
            out = self.model.norm(hidden)
        else:
            out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if return_hidden:
            return out, hidden
        return out

    @property
    def mtp_capability(self) -> MTPCapability:
        """Return an immutable eligibility snapshot for native MTP dispatch."""
        if not hasattr(self, "mtp"):
            return MTPCapability(False, "native_mtp_head_not_configured", 0)
        if self.model.pipeline_size != 1:
            return MTPCapability(
                False,
                "native_mtp_pipeline_parallelism_unsupported",
                self.args.mtp_num_hidden_layers,
            )
        if not self._mtp_weight_keys_validated:
            return MTPCapability(
                False,
                "native_mtp_weights_not_validated",
                self.args.mtp_num_hidden_layers,
            )
        if not self._mtp_weights_loaded:
            return MTPCapability(
                False,
                "native_mtp_weights_not_loaded",
                self.args.mtp_num_hidden_layers,
            )
        return MTPCapability(True, "supported", self.args.mtp_num_hidden_layers)

    @property
    def supports_mtp(self) -> bool:
        return self.mtp_capability.supported

    def _require_mtp(self) -> None:
        capability = self.mtp_capability
        if not capability.supported:
            raise RuntimeError(capability.reason)

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: List[Any],
    ) -> mx.array:
        """Project an MTP step through the backbone's shared output head."""
        self._require_mtp()
        states = self.mtp(
            hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(states)
        return self.lm_head(states)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def make_mtp_cache(self) -> List[KVCache]:
        self._require_mtp()
        return [KVCache() for _ in self.mtp.layers]

    def make_mtp_request_cache(self, *, prompt_cache: Optional[Any] = None):
        """Return fresh, request-local native-MTP cache state.

        Generic prefix caches are intentionally rejected here: they cannot yet
        establish exact recurrent/MTP-head alignment.  The generation layer
        will use this structured owner instead of positional cache splitting.
        """

        return NativeMTPRequestCache.create(self, prompt_cache=prompt_cache)

    @staticmethod
    def _weight_handshake(weights):
        """Identify the exact sanitized key/array mapping handed to the loader."""
        try:
            items = list(weights.items())
        except AttributeError:
            items = list(weights)
        return tuple(sorted((key, id(value)) for key, value in items))

    def _consume_mtp_load_handshake(self, weights) -> bool:
        pending = self._mtp_pending_load_handshake
        self._mtp_pending_load_handshake = None
        if pending is None:
            return False
        return pending == self._weight_handshake(weights)

    def _validate_mtp_load_leaves(self, weights) -> None:
        """Validate MTP arrays against the actual post-quantization module tree."""
        supplied = {
            key: value
            for key, value in weights
            if key.startswith("language_model.mtp.")
        }
        expected = {
            f"language_model.mtp.{key}": value
            for key, value in tree_flatten(self.mtp.parameters())
        }
        if supplied.keys() != expected.keys():
            missing = sorted(expected.keys() - supplied.keys())
            unexpected = sorted(supplied.keys() - expected.keys())
            details = []
            if missing:
                details.append("missing " + ", ".join(missing[:3]))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected[:3]))
            raise ValueError(
                "Native Qwen MTP load mapping does not match the runtime head: "
                + "; ".join(details)
            )

        mismatches = []
        for key, expected_value in expected.items():
            supplied_value = supplied[key]
            same_type = type(supplied_value) is type(expected_value)
            same_shape = getattr(supplied_value, "shape", None) == getattr(
                expected_value, "shape", None
            )
            if not same_type or not same_shape:
                mismatches.append(
                    f"{key} expected {type(expected_value).__name__}"
                    f"{tuple(expected_value.shape)} got "
                    f"{type(supplied_value).__name__}"
                    f"{tuple(getattr(supplied_value, 'shape', ()))}"
                )
        if mismatches:
            raise ValueError(
                "Native Qwen MTP load leaf type/shape mismatch: "
                + "; ".join(mismatches[:3])
            )

    def load_weights(self, weights, strict=True):
        if isinstance(weights, (str, bytes)) or hasattr(weights, "__fspath__"):
            matches_pending = False
            load_weights = weights
        else:
            load_weights = (
                list(weights.items()) if hasattr(weights, "items") else list(weights)
            )
            matches_pending = self._consume_mtp_load_handshake(load_weights)
            if matches_pending:
                self._validate_mtp_load_leaves(load_weights)
        result = super().load_weights(load_weights, strict=strict)
        if hasattr(self, "mtp") and self._mtp_weight_keys_validated and matches_pending:
            self._mtp_weights_loaded = True
        return result

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        should_shift_norm_weights = has_unsanitized_conv1d
        if not hasattr(self, "mtp"):
            # A base checkpoint may carry a draft head intended for a different
            # loader.  Never accidentally expose it on a no-head model.
            weights = {k: v for k, v in weights.items() if "mtp." not in k}
        else:
            # Every sanitation attempt starts a fresh, one-shot load handshake.
            # A later invalid checkpoint must invalidate any earlier pending map.
            self._mtp_weight_keys_validated = False
            self._mtp_weights_loaded = False
            self._mtp_pending_load_handshake = None
            expected = {
                f"language_model.mtp.{name}"
                for name, _ in tree_flatten(self.mtp.parameters())
            }
            quantizable = {
                f"language_model.mtp.{name}"
                for name, module in tree_flatten(
                    self.mtp.leaf_modules(), is_leaf=nn.Module.is_module
                )
                if hasattr(module, "to_quantized")
            }
            supplied = {key for key in weights if key.startswith("language_model.mtp.")}
            quant_auxiliary = set()
            partial_quantization = []
            for weight_key in (
                key
                for key in expected
                if key.endswith(".weight")
                and key.removesuffix(".weight") in quantizable
            ):
                prefix = weight_key.removesuffix(".weight")
                scales_key = f"{prefix}.scales"
                biases_key = f"{prefix}.biases"
                has_scales = scales_key in supplied
                has_biases = biases_key in supplied
                if has_scales != has_biases:
                    partial_quantization.append(prefix)
                elif has_scales:
                    quant_auxiliary.update((scales_key, biases_key))

            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected - quant_auxiliary)
            if missing or unexpected or partial_quantization:
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing[:3]))
                if unexpected:
                    details.append("unexpected " + ", ".join(unexpected[:3]))
                if partial_quantization:
                    details.append(
                        "incomplete quantized triplet "
                        + ", ".join(sorted(partial_quantization)[:3])
                    )
                raise ValueError(
                    "Native Qwen MTP weights do not match the configured head: "
                    + "; ".join(details)
                )
            self._mtp_weight_keys_validated = True
            self._mtp_weights_loaded = False

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        if hasattr(self, "mtp"):
            self._mtp_pending_load_handshake = self._weight_handshake(weights)
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0 and self.args.mtp_num_hidden_layers <= 0:
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
        return_hidden: bool = False,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model.mtp."):
                # The VLM wrapper's MTP head belongs beside language_model.model
                # rather than underneath the backbone ``model`` module.
                key = key.replace("model.language_model.", "language_model.", 1)
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        if self.language_model.args.mtp_num_hidden_layers:
            raise RuntimeError(
                "Native Qwen MTP does not support tensor/distributed sharding"
            )
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
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    def load_weights(self, weights, strict=True):
        if isinstance(weights, (str, bytes)) or hasattr(weights, "__fspath__"):
            matches_pending = False
            load_weights = weights
        else:
            load_weights = (
                list(weights.items()) if hasattr(weights, "items") else list(weights)
            )
            matches_pending = self.language_model._consume_mtp_load_handshake(
                load_weights
            )
            if matches_pending:
                self.language_model._validate_mtp_load_leaves(load_weights)
        result = super().load_weights(load_weights, strict=strict)
        if self.language_model._mtp_weight_keys_validated and matches_pending:
            self.language_model._mtp_weights_loaded = True
        return result

    @property
    def mtp_capability(self) -> MTPCapability:
        return self.language_model.mtp_capability

    @property
    def supports_mtp(self) -> bool:
        return self.language_model.supports_mtp

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: List[Any],
    ) -> mx.array:
        return self.language_model.mtp_forward(hidden_states, next_token_ids, mtp_cache)

    def make_mtp_cache(self) -> List[KVCache]:
        return self.language_model.make_mtp_cache()

    def make_mtp_request_cache(self, *, prompt_cache: Optional[Any] = None):
        return self.language_model.make_mtp_request_cache(prompt_cache=prompt_cache)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
