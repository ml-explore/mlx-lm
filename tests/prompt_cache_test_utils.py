# Copyright © 2024 Apple Inc.

import mlx.core as mx

from mlx_lm.models.cache import RotatingKVCache


def make_tiny_step3p5_model():
    from mlx_lm.models import step3p5

    # Keep this config minimal and centralized so schema churn in one place
    # does not ripple through multiple cache behavior assertions.
    args = step3p5.ModelArgs.from_dict(
        {
            "model_type": "step3p5",
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "vocab_size": 256,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "rms_norm_eps": 1e-5,
            "rope_theta": [10000.0, 10000.0, 10000.0, 10000.0],
            "sliding_window": 4,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "partial_rotary_factors": [1.0, 1.0, 1.0, 1.0],
            "attention_other_setting": {
                "num_attention_heads": 4,
                "num_attention_groups": 2,
            },
            "use_head_wise_attn_gate": True,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 128,
            "share_expert_dim": 128,
            "moe_layers_enum": "1,2,3",
        }
    )
    return step3p5.Model(args)


def build_real_rotating_cache(*, max_size=4, total_tokens=4):
    cache = RotatingKVCache(max_size=max_size)
    kv = mx.arange(total_tokens, dtype=mx.float32).reshape(1, 1, total_tokens, 1)
    cache.update_and_fetch(kv, kv)
    mx.eval(cache.keys, cache.values)
    return cache


def snapshot_cache_arrays(cache):
    keys = mx.array(cache.keys) if cache.keys is not None else None
    values = mx.array(cache.values) if cache.values is not None else None
    if keys is not None:
        mx.eval(keys)
    if values is not None:
        mx.eval(values)
    return keys, values


class RewindRecorderLayer:
    def __init__(
        self,
        *,
        max_rewind=4,
        rewind_result=True,
        offset=None,
        rewind_calls=None,
        can_rewind_calls=None,
    ):
        self.max_rewind = max_rewind
        self.rewind_result = rewind_result
        self.rewind_calls = rewind_calls if rewind_calls is not None else []
        self.can_rewind_calls = can_rewind_calls if can_rewind_calls is not None else []
        self.offset = max_rewind if offset is None else offset

    @property
    def nbytes(self):
        return 1

    def can_rewind(self, n):
        self.can_rewind_calls.append(n)
        return n <= self.max_rewind

    def rewind(self, n):
        self.rewind_calls.append(n)
        if self.rewind_result:
            self.offset = max(0, self.offset - n)
        return self.rewind_result

    def __deepcopy__(self, memo):
        return type(self)(
            max_rewind=self.max_rewind,
            rewind_result=self.rewind_result,
            offset=self.offset,
            rewind_calls=self.rewind_calls,
            can_rewind_calls=self.can_rewind_calls,
        )


class LegacyTrimLayer:
    def __init__(
        self,
        *,
        offset=4,
        trim_shortfall=0,
        trim_calls=None,
        deepcopy_calls=None,
    ):
        self.offset = offset
        self.trim_shortfall = trim_shortfall
        self.trim_calls = trim_calls if trim_calls is not None else []
        self.deepcopy_calls = deepcopy_calls if deepcopy_calls is not None else []

    @property
    def nbytes(self):
        return 1

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trim_calls.append(n)
        trimmed = max(0, n - self.trim_shortfall)
        self.offset = max(0, self.offset - trimmed)
        return trimmed

    def __deepcopy__(self, memo):
        self.deepcopy_calls.append("deepcopy")
        return type(self)(
            offset=self.offset,
            trim_shortfall=self.trim_shortfall,
            trim_calls=self.trim_calls,
            deepcopy_calls=self.deepcopy_calls,
        )


class UnknownNonTrimmableLayer:
    @property
    def nbytes(self):
        return 1


class UnknownNonTrimmableNoDeepcopy:
    @property
    def nbytes(self):
        return 1

    def __deepcopy__(self, memo):
        raise AssertionError(
            "deepcopy should be skipped for unknown non-trimmable layers"
        )


class UnknownLayerWithoutLegacyHooks:
    @property
    def nbytes(self):
        return 1

    def __deepcopy__(self, memo):
        raise AssertionError(
            "deepcopy should be skipped when layer lacks rewind and trim contracts"
        )


class DeepcopyShouldNotRunLayer:
    @property
    def nbytes(self):
        return 1

    def can_rewind(self, n):
        return True

    def rewind(self, n):
        return True

    def __deepcopy__(self, memo):
        raise AssertionError("deepcopy should be skipped on known-safe miss")
