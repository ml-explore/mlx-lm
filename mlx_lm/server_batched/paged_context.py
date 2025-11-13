# ABOUTME: Provides context managers for paged-attention runtime state.
# ABOUTME: Tracks current transformer layer and batch view via ContextVar.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional, Sequence

CURRENT_LAYER: ContextVar[Optional[int]] = ContextVar(
    "mlx_lm_paged_layer", default=None
)
CURRENT_BATCH_VIEW: ContextVar[Optional[Any]] = ContextVar(
    "mlx_lm_paged_batch_view", default=None
)


@contextmanager
def layer_scope(layer_idx: int) -> Iterator[None]:
    """Temporarily set the current transformer layer index."""
    token = CURRENT_LAYER.set(layer_idx)
    try:
        yield
    finally:
        CURRENT_LAYER.reset(token)


@contextmanager
def batch_view_scope(view: Optional[Any]) -> Iterator[None]:
    """Temporarily set the batch-level paged attention view."""
    if view is None:
        yield
        return
    token = CURRENT_BATCH_VIEW.set(view)
    try:
        yield
    finally:
        CURRENT_BATCH_VIEW.reset(token)


def current_layer() -> Optional[int]:
    return CURRENT_LAYER.get(None)


def current_batch_view() -> Optional[Any]:
    return CURRENT_BATCH_VIEW.get(None)


def _wrap_fn_with_layer_scope(obj: Any, fn_name: str, layer_idx: int) -> bool:
    """Wrap obj.fn_name to push layer scope; returns True when wrapped."""
    if obj is None:
        return False
    if getattr(obj, "_paged_wrapped", False):
        return True

    cls = obj.__class__
    method = getattr(cls, fn_name, None)
    if method is None:
        return False

    def scoped_call(self, *args, __orig=method, __idx=layer_idx, **kwargs):
        with layer_scope(__idx):
            return __orig(self, *args, **kwargs)

    scoped_cls = type(
        f"{cls.__name__}_PagedLayer_{layer_idx}",
        (cls,),
        {fn_name: scoped_call},
    )
    obj.__class__ = scoped_cls
    setattr(obj, "_paged_wrapped", True)
    return True


def _iter_layer_lists(model: Any) -> Sequence:
    """Return the first iterable of layers discovered on the model."""
    seen = set()
    queue = [model]
    while queue:
        current = queue.pop(0)
        key = id(current)
        if key in seen:
            continue
        seen.add(key)
        layers = getattr(current, "layers", None)
        if isinstance(layers, (list, tuple)) and layers:
            return layers
        for attr in ("model", "transformer", "llm", "decoder"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append(child)
    return []


def wrap_attention_layers(model: Any) -> None:
    """Ensure every attention module enters layer_scope regardless of call path."""
    layers = _iter_layer_lists(model)
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            continue
        # Wrap __call__ when defined; fall back to forward for alternate modules.
        wrapped = _wrap_fn_with_layer_scope(attn, "__call__", idx)
        if not wrapped:
            wrapped = _wrap_fn_with_layer_scope(attn, "forward", idx)
        if not wrapped:
            continue


__all__ = [
    "layer_scope",
    "batch_view_scope",
    "current_layer",
    "current_batch_view",
    "wrap_attention_layers",
]
