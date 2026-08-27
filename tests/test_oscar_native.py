# Copyright © 2026 MLX Contributors
# SPDX-License-Identifier: MIT

import copy
import json

import mlx.core as mx
import pytest

from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import (
    CacheList,
    KVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.models.oscar import (
    OscarKVCache,
    hadamard,
    int2_dequantize,
    int2_quantize,
    load_rotations,
    relative_error,
)
from mlx_lm.oscar_calibration import calibrate_rotations, save_rotations


def _random(shape, dtype=mx.float32):
    return mx.random.normal(shape).astype(dtype)


def test_int2_codec_packs_four_values_per_byte_and_round_trips():
    values = _random((2, 3, 5, 16))
    packed, scales, zeros = int2_quantize(values, group_size=8)
    assert packed.dtype == mx.uint8
    assert packed.shape == (2, 3, 5, 4)
    assert scales.shape == zeros.shape == (2, 3, 5, 2)
    restored = int2_dequantize(packed, scales, zeros, group_size=8)
    mx.eval(restored)
    assert restored.shape == values.shape
    assert relative_error(values, restored) < 0.8


def test_rotation_loader_supports_per_layer_and_shared_files(tmp_path):
    matrices = {f"layer_{i}": hadamard(8) for i in range(2)}
    mx.save_safetensors(str(tmp_path / "k_rotation.safetensors"), matrices)
    mx.save_safetensors(str(tmp_path / "v_rotation.safetensors"), matrices)
    loaded = load_rotations(tmp_path, 2)
    assert len(loaded.rK) == len(loaded.rV) == 2
    assert loaded.rK[0].shape == (8, 8)

    shared = tmp_path / "shared"
    shared.mkdir()
    mx.save_safetensors(str(shared / "rotation.safetensors"), {"rotation": hadamard(8)})
    shared_loaded = load_rotations(shared, 3)
    assert len(shared_loaded.rK) == len(shared_loaded.rV) == 3

    sparse = tmp_path / "sparse"
    sparse.mkdir()
    mx.save_safetensors(
        str(sparse / "k_rotation.safetensors"), {"layer_3": hadamard(8)}
    )
    mx.save_safetensors(
        str(sparse / "v_rotation.safetensors"), {"layer_3": hadamard(8)}
    )
    assert load_rotations(sparse, 1).rK[0].shape == (8, 8)


def test_oscar_cache_ages_recent_tokens_once_and_returns_full_view():
    keys = _random((1, 2, 12, 8), mx.bfloat16)
    values = _random((1, 2, 12, 8), mx.bfloat16)
    cache = OscarKVCache(group_size=4, sink_tokens=2, recent_tokens=3)
    output = cache.update_and_fetch(keys, values)
    mx.eval(output, cache.state)
    assert cache.offset == 12
    assert output[0].shape == keys.shape
    assert cache.state[0].shape[2] == 2
    assert cache.state[2].shape[2] == 7
    assert cache.state[8].shape[2] == 3

    packed_before = cache.state[2]
    cache.update_and_fetch(keys[..., :1, :], values[..., :1, :])
    mx.eval(cache.state)
    assert cache.offset == 13
    assert cache.state[2].shape[2] >= packed_before.shape[2]

    cloned = copy.deepcopy(cache)
    assert isinstance(cloned, OscarKVCache)
    assert cloned.offset == cache.offset


def test_native_prompt_cache_round_trip_restores_oscar_state_and_requires_rotation_binding(
    tmp_path,
):
    values = _random((1, 1, 12, 8), mx.bfloat16)
    rotation = hadamard(8)
    cache = OscarKVCache(
        rK=rotation,
        rV=rotation,
        group_size=4,
        sink_tokens=2,
        recent_tokens=3,
    )
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    path = tmp_path / "prompt.safetensors"
    save_prompt_cache(str(path), [cache], {"model": "fixture"})
    restored, metadata = load_prompt_cache(str(path), return_metadata=True)
    restored = restored[0]
    assert isinstance(restored, OscarKVCache)
    assert restored.offset == cache.offset
    assert metadata == {"model": "fixture"}
    with pytest.raises(ValueError, match="rotation fingerprint"):
        restored._reconstruct()
    restored.bind_rotations(rotation, rotation)
    actual = restored._reconstruct()
    mx.eval(actual)
    assert actual[0].shape == values.shape


def test_state_rejects_partial_packed_tiers_and_mismatched_meta():
    values = _random((1, 1, 12, 8), mx.bfloat16)
    cache = OscarKVCache(group_size=4, sink_tokens=2, recent_tokens=3)
    cache.update_and_fetch(values, values)
    state = list(cache.state)
    state[3] = None
    with pytest.raises(ValueError, match="all six packed"):
        OscarKVCache.from_state(state, cache.meta_state)
    with pytest.raises(ValueError, match="does not match state length"):
        OscarKVCache.from_state(
            cache.serialized_state(),
            json.dumps({"offset": 11, "group_size": 4}),
        )


def test_trim_drops_newest_tokens_and_reappend_returns_to_same_offset():
    values = _random((1, 1, 20, 8), mx.bfloat16)
    cache = OscarKVCache(group_size=4, sink_tokens=2, recent_tokens=3)
    cache.update_and_fetch(values, values)
    assert cache.offset == 20
    assert cache.trim(4) == 4
    assert cache.offset == 16
    out = cache.update_and_fetch(values[..., 16:20, :], values[..., 16:20, :])
    mx.eval(out)
    assert cache.offset == 20
    assert out[0].shape[2] == 20


def test_bounded_attention_uses_native_hook_for_gqa_decode():
    B, Hq, Hkv, D = 1, 4, 2, 128
    keys = _random((B, Hkv, 300, D), mx.bfloat16)
    values = _random((B, Hkv, 300, D), mx.bfloat16)
    query = _random((B, Hq, 1, D), mx.bfloat16)
    cache = OscarKVCache(
        group_size=128,
        sink_tokens=64,
        recent_tokens=64,
        bounded_attention=True,
    )
    cache.update_and_fetch(keys, values)
    assert cache.update_and_fetch(keys[..., :1, :], values[..., :1, :]) == (None, None)
    output = scaled_dot_product_attention(
        query, None, None, cache=cache, scale=D**-0.5, mask=None
    )
    mx.eval(output)
    assert output.shape == query.shape
    assert output.dtype == query.dtype


def test_bounded_attention_falls_back_for_small_non_native_group_kernel():
    D = 8
    keys = _random((1, 2, 8, D), mx.bfloat16)
    query = _random((1, 2, 1, D), mx.bfloat16)
    cache = OscarKVCache(group_size=4, sink_tokens=2, recent_tokens=2, bounded_attention=True)
    cache.update_and_fetch(keys, keys)
    cache.update_and_fetch(query, query)
    output = scaled_dot_product_attention(
        query, None, None, cache=cache, scale=D**-0.5, mask=None
    )
    mx.eval(output)
    assert output.shape == query.shape


def test_bounded_attention_runs_through_a_native_gqa_model():
    from mlx_lm.models.llama import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )
    cache = make_prompt_cache(
        model,
        oscar_config={
            "group_size": 8,
            "sink_tokens": 2,
            "recent_tokens": 2,
            "bounded_attention": True,
        },
    )
    logits = model(mx.array([[1, 2, 3, 4, 5, 6]]), cache=cache)
    next_logits = model(mx.array([[7]]), cache=cache)
    mx.eval(logits, next_logits, [c.state for c in cache])
    assert logits.shape == (1, 6, 64)
    assert next_logits.shape == (1, 1, 64)
    assert [c.offset for c in cache] == [7, 7]


def test_explicit_oscar_prompt_cache_is_opt_in_and_preserves_nested_cache_types():
    class Model:
        def make_cache(self):
            return [KVCache(), CacheList(KVCache(), KVCache())]

    plain = make_prompt_cache(Model())
    assert type(plain[0]) is KVCache
    opted = make_prompt_cache(
        Model(),
        oscar_config={"group_size": 4, "sink_tokens": 2, "recent_tokens": 2},
    )
    assert isinstance(opted[0], OscarKVCache)
    assert isinstance(opted[1].caches[0], OscarKVCache)
    assert isinstance(opted[1].caches[1], OscarKVCache)


def test_offline_calibration_writes_portable_rotation_artifacts(tmp_path):
    samples = _random((2, 2, 16, 8))
    rotations = calibrate_rotations(samples, samples)
    assert rotations.rK[0].shape == (2, 8, 8)
    save_rotations(tmp_path, rotations)
    assert (tmp_path / "k_rotation.safetensors").exists()
    assert (tmp_path / "v_rotation.safetensors").exists()
    manifest = json.loads((tmp_path / "metadata.json").read_text())
    assert manifest["algorithm"] == "FutureMLS-Lab OSCAR"
    assert "no source copied" in manifest["provenance"]["sglang"]
    loaded = load_rotations(tmp_path, 1)
    assert loaded.rV[0].shape == (2, 8, 8)
