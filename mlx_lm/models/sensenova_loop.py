"""MLX-LM adapter for SenseNova-Loop (model_type: sensenova_loop).

Backbone is Qwen3.5-text (GatedDeltaNet linear attention + full attention).
This adapter adds:
  - Loop execution plan (layers [start,end) run count times with shared weights)
  - Two-stream mHC routing (MHCRouter + MHCDecoderLayer)

Supports both the Preview and the released SenseNova-Loop-9B checkpoints
(loop_settings [4, 28) x2 with two-stream mHC over the same range).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .qwen3_5 import (
    GatedDeltaNet,
    Qwen3_5TextModel,
    TextModelArgs as Qwen3_5TextModelArgs,
    Attention,
    MLP,
)
from .base import BaseModelArgs, create_attention_mask, create_ssm_mask
from .cache import ArraysCache, KVCache


# ---------------------------------------------------------------------------
# Sinkhorn
# ---------------------------------------------------------------------------

def _sinkhorn(logits: mx.array, num_iters: int) -> mx.array:
    for _ in range(num_iters):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


# ---------------------------------------------------------------------------
# MHC router
# ---------------------------------------------------------------------------

class MHCRouter(nn.Module):
    """Two-stream manifold-constrained hyper-connection router."""

    def __init__(self, hidden_size: int, streams: int, iterations: int):
        super().__init__()
        self.streams = streams
        self.iterations = iterations
        out_features = 2 * streams + streams * streams
        self.weight = mx.zeros((out_features, streams * hidden_size))
        self.bias = mx.zeros((out_features,))

    def __call__(self, residual: mx.array) -> tuple:
        # residual: [B, T, streams, hidden]
        B, T, s, H = residual.shape
        flat = mx.reshape(residual, (B, T, s * H))
        normed = flat * mx.rsqrt(flat.square().mean(-1, keepdims=True) + 1e-6)
        logits = normed @ self.weight.T + self.bias
        pre = mx.sigmoid(logits[..., :self.streams])
        post = 2.0 * mx.sigmoid(logits[..., self.streams : 2 * self.streams])
        mix_logits = logits[..., 2 * self.streams :].reshape(
            *logits.shape[:-1], self.streams, self.streams
        )
        mix = _sinkhorn(mix_logits, self.iterations)
        branch = mx.matmul(pre[..., None, :], residual).squeeze(-2)
        return branch, post, mix

    @staticmethod
    def merge(residual: mx.array, branch: mx.array, post: mx.array, mix: mx.array) -> mx.array:
        mixed = mx.matmul(mix, residual)
        update = post[..., None] * branch[..., None, :]
        return mixed + update


# ---------------------------------------------------------------------------
# mHC decoder layer
# ---------------------------------------------------------------------------

class MHCDecoderLayer(nn.Module):
    """Qwen3_5 DecoderLayer wrapped with two-stream mHC."""

    def __init__(self, base_layer: nn.Module, hidden_size: int, layer_idx: int, mhc_settings: dict):
        super().__init__()
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.mlp = base_layer.mlp
        self.is_linear = base_layer.is_linear
        if base_layer.is_linear:
            self.linear_attn = base_layer.linear_attn
        else:
            self.self_attn = base_layer.self_attn
        self.layer_idx = layer_idx

        self.mhc_settings = mhc_settings
        self.mhc_attn = MHCRouter(hidden_size, mhc_settings["streams"], mhc_settings["iterations"])
        self.mhc_ffn = MHCRouter(hidden_size, mhc_settings["streams"], mhc_settings["iterations"])

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        s = self.mhc_settings["streams"]
        H = hidden_states.shape[-1]
        # Expand to streams: [B, T, hidden] -> [B, T, streams, hidden]
        expanded = mx.expand_dims(hidden_states, axis=-2)
        target_shape = (expanded.shape[0], expanded.shape[1], s, H)
        residual = mx.broadcast_to(expanded, target_shape)

        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(branch)
        if self.is_linear:
            branch = self.linear_attn(branch, mask, cache)
        else:
            branch = self.self_attn(branch, mask, cache)
        residual = MHCRouter.merge(residual, branch, post, mix)

        branch, post, mix = self.mhc_ffn(residual)
        branch = self.post_attention_layernorm(branch)
        branch = self.mlp(branch)
        out = MHCRouter.merge(residual, branch, post, mix)
        # Collapse streams back to hidden dimension
        return mx.mean(out, axis=-2)


# ---------------------------------------------------------------------------
# Text model with loop execution (mirrors qwen3_5 TextModel structure)
# ---------------------------------------------------------------------------

class TextModel(nn.Module):
    """Wraps Qwen3_5TextModel with loop execution and mHC."""

    def __init__(self, args: dict, loop_settings: dict = None, mhc_settings: dict = None):
        super().__init__()
        qwen_args = Qwen3_5TextModelArgs.from_dict(args)
        self.args = qwen_args
        self.model_type = qwen_args.model_type
        self.model = Qwen3_5TextModel(qwen_args)
        self._loop_settings = loop_settings or {}
        self._mhc_settings = mhc_settings or {}

        if not qwen_args.tie_word_embeddings:
            self.lm_head = nn.Linear(qwen_args.hidden_size, qwen_args.vocab_size, bias=False)

        # Wrap looped layers with mHC
        self._wrap_looped_layers()

    def _wrap_looped_layers(self):
        start = self._mhc_settings.get("start", 0)
        end = self._mhc_settings.get("end", 0)
        new_layers = list(self.model.layers)
        for i in range(start, end):
            old_layer = self.model.layers[i]
            new_layer = MHCDecoderLayer(old_layer, self.args.hidden_size, i, self._mhc_settings)
            new_layers[i] = new_layer
        self.model.layers = new_layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.model.embed_tokens(inputs)

        if cache is None:
            plan = self._build_execution_plan()
            cache = [None] * len(plan)

        fa_mask = None
        ssm_mask = None
        if self.model.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.model.fa_idx])
        if self.model.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.model.ssm_idx])

        plan = self._build_execution_plan()

        for physical_idx, virtual_idx in plan:
            layer = self.model.layers[physical_idx]
            c = cache[virtual_idx] if cache else None
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        hidden_states = self.model.norm(hidden_states)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        else:
            return self.lm_head(hidden_states)

    def _build_execution_plan(self) -> list:
        loop = self._loop_settings
        num_layers = self.args.num_hidden_layers
        start = loop.get("start", 0)
        end = loop.get("end", num_layers)
        count = loop.get("eval_count", 1)
        plan = list(range(start)) + [i for _ in range(count) for i in range(start, end)] + list(range(end, num_layers))
        virtual = 0
        result = []
        for physical in plan:
            result.append((physical, virtual))
            virtual += 1
        return result

    def make_cache(self):
        plan = self._build_execution_plan()
        caches = []
        for physical_idx, _ in plan:
            layer = self.model.layers[physical_idx]
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(KVCache())
        return caches

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers

    @property
    def norm(self):
        return self.model.norm

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        # The released checkpoint carries a ``loop_gate.logits`` parameter
        # that the reference implementation never reads and this module does
        # not declare; without this filter the whole load fails with
        # "Received 1 parameters not in model".
        weights = {k: v for k, v in weights.items() if "loop_gate" not in k}

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
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
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


# ---------------------------------------------------------------------------
# Model args and Model
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "sensenova_loop"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    loop_settings: dict = field(default_factory=dict)
    mhc: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.text_config:
            tc = self.text_config
            if "model_type" not in tc:
                tc["model_type"] = "qwen3_5_text"


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(
            args.text_config,
            loop_settings=args.loop_settings,
            mhc_settings=args.mhc,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(inputs, cache=cache, input_embeddings=input_embeddings)

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)
