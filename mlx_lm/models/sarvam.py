from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    first_k_dense_replace: int = 1
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    moe_intermediate_size: int = 1024
    moe_router_enable_expert_bias: bool = True
    moe_shared_expert_intermediate_size: int = 1024
    n_group: int = 1
    norm_topk_prob: bool = True
    num_experts: int = 128
    num_experts_per_tok: int = 6
    num_key_value_heads: Optional[int] = None
    num_shared_experts: int = 1
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_theta: float = 8e6
    routed_scaling_factor: float = 2.5
    score_function: str = "sigmoid"
    tie_word_embeddings: bool = False
    topk_group: int = 1
    use_qk_norm: bool = True


class SarvamAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        if args.head_dim is not None:
            self.head_dim = args.head_dim
        else:
            self.head_dim = args.hidden_size // args.num_attention_heads
        if args.num_key_value_heads is not None:
            self.num_key_value_heads = args.num_key_value_heads
        else:
            self.num_key_value_heads = args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        bsz, seq_len, dim = x.shape
        qkv = self.query_key_value(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)
        queries = queries.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)
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
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.dense(output)


class SarvamMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.swiglu(self.gate_proj(x), self.up_proj(x)))

    @staticmethod
    @partial(mx.compile, shapeless=True)
    def swiglu(x: mx.array, y: mx.array) -> mx.array:
        return nn.silu(x) * y


class SarvamMoEGate(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.score_function = args.score_function

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        in_type = x.dtype
        gates = x @ self.weight.T
        orig_scores, scores = self.gate_score(
            gates=gates,
            bias=self.expert_bias,
            use_sigmoid=self.score_function == "sigmoid",
        )
        if self.n_group > 1:
            scores = mx.unflatten(scores, axis=-1, shape=(self.n_group, -1))
            group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            scores = mx.put_along_axis(
                scores,
                mx.stop_gradient(group_idx),
                mx.array(0.0, scores.dtype),
                axis=-2,
            )
            scores = mx.flatten(scores, -2, -1)
        inds = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
        scores = scores * self.routed_scaling_factor
        return inds, scores.astype(in_type)

    @staticmethod
    @partial(mx.compile, shapeless=True)
    def gate_score(
        gates: mx.array, bias: Optional[mx.array], use_sigmoid: bool
    ) -> Tuple[mx.array, mx.array]:
        scores = (
            mx.sigmoid(gates.astype(mx.float32))
            if use_sigmoid
            else mx.softmax(gates.astype(mx.float32), axis=-1)
        )
        return scores, scores + bias if bias is not None else scores


class SarvamSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate = SarvamMoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        if args.num_shared_experts is not None and args.num_shared_experts > 0:
            self.shared_experts: Optional[SarvamMLP] = SarvamMLP(
                hidden_size=args.hidden_size,
                intermediate_size=args.moe_shared_expert_intermediate_size
                * args.num_shared_experts,
            )
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        out = self.switch_mlp(x, inds)
        out = self.aggregate_expert_outputs(out, scores)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out

    @staticmethod
    @partial(mx.compile, shapeless=True)
    def aggregate_expert_outputs(
        expert_outputs: mx.array, scores: mx.array
    ) -> mx.array:
        return (
            (expert_outputs * scores[..., None])
            .sum(axis=-2)
            .astype(expert_outputs.dtype)
        )


class SarvamDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.attention = SarvamAttention(args)
        if args.num_experts is not None and layer_idx >= args.first_k_dense_replace:
            self.mlp: nn.Module = SarvamSparseMoeBlock(args)
        else:
            self.mlp = SarvamMLP(
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
            )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.input_layernorm(x)
        gqa = self.attention(r, mask, cache)
        h = x + gqa
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class SarvamModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            SarvamDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = create_attention_mask(h, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            h = layer(h, mask, layer_cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SarvamModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx in range(
            self.args.first_k_dense_replace, self.args.num_hidden_layers
        ):
            prefix = f"model.layers.{layer_idx}"
            for m in ("gate_proj", "down_proj", "up_proj"):
                for k in ("weight", "scales", "biases"):
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
                    elif f"{prefix}.mlp.experts.{m}.{k}" in weights:
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = weights.pop(
                            f"{prefix}.mlp.experts.{m}.{k}"
                        )
        return weights

    @property
    def quant_predicate(
        self,
    ) -> Callable[[str, nn.Module], Union[bool, Dict[str, Any]]]:
        def _predicate(path: str, _: nn.Module) -> bool | Dict[str, Any]:
            if "word_embeddings" in path:
                return False
            # LM head is the final projection to logits. 4-bit error here directly flips token
            # choices (e.g. '_' vs space).
            if "lm_head" in path:
                return False
            # Shared experts are in the critical path for every token. Use 8-bit.
            if "shared_experts" in path:
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return _predicate

    @property
    def layers(self) -> list[SarvamDecoderLayer]:
        return self.model.layers
