# Copyright © 2023-2024 Apple Inc.

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

def warn_discarded_weights(
    n_discarded: int,
    feature_name: str = "MTP",
    model_name: Optional[str] = None,
) -> None:
    """Emit a one-time warning when feature-specific weights are dropped at loading time.

    Some model checkpoints (e.g., Multi-Token Prediction variants) ship with
    additional weight tensors that are discarded during loading because mlx-lm
    does not currently use them at runtime. This warning surfaces the
    discrepancy at load time so downstream benchmarks can interpret results
    correctly.

    Args:
        n_discarded: Number of weight tensors filtered out by the caller.
          The warning is only emitted when this is +ive.
        feature_name: Name of the discarded feature.
          Defaults to "MTP".
        model_name: Optional.
    """
    if n_discarded <= 0:
        return
    suffix = f" for model '{model_name}'" if model_name else ""
    warnings.warn(
        f"Loaded {n_discarded} {feature_name} weight tensor(s){suffix}, "
        f"discarded at runtime — mlx-lm does not currently support "
        f"{feature_name}. The model will run as a standard autoregressive "
        f"model.",
        UserWarning,
        stacklevel=2,  # warning come from the model file, not from base.py
    )
