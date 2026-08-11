# Copyright © 2026 Apple Inc.
"""Synthetic model-only coverage for Qwen native MTP capability handling."""

import importlib
import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

HIDDEN = 32
INTERMEDIATE = 64
HEAD_DIM = 8
NUM_HEADS = 4
NUM_KV_HEADS = 2
NUM_EXPERTS = 2
MOE_INTERMEDIATE = 32
SHARED_INTERMEDIATE = 64


def _config(*, moe=False, mtp_layers=1, tie_word_embeddings=True):
    config = {
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_moe" if moe else "qwen3_5",
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": 2,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "vocab_size": 64,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 3,
            "full_attention_interval": 2,
            "tie_word_embeddings": tie_word_embeddings,
            "rms_norm_eps": 1e-5,
            "head_dim": HEAD_DIM,
            "rope_theta": 1000.0,
            "partial_rotary_factor": 0.5,
            "max_position_embeddings": 128,
            "mtp_num_hidden_layers": mtp_layers,
        },
    }
    if moe:
        config["text_config"].update(
            {
                "num_experts": NUM_EXPERTS,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": SHARED_INTERMEDIATE,
                "moe_intermediate_size": MOE_INTERMEDIATE,
            }
        )
    return config


def _model(*, moe=False, mtp_layers=1, tie_word_embeddings=True):
    module_name = "mlx_lm.models.qwen3_5_moe" if moe else "mlx_lm.models.qwen3_5"
    module = importlib.import_module(module_name)
    model = module.Model(
        module.ModelArgs.from_dict(
            _config(
                moe=moe,
                mtp_layers=mtp_layers,
                tie_word_embeddings=tie_word_embeddings,
            )
        )
    )
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def _fixture(shape, seed):
    size = math.prod(shape)
    return (mx.arange(size, dtype=mx.float32).reshape(shape) + seed) / (size + seed)


def _canonical_mtp_weights(*, moe=False, mtp_layers=1):
    """Build checkpoint keys from the published Qwen schema, not model.parameters."""
    weights = {
        "language_model.mtp.pre_fc_norm_hidden.weight": _fixture((HIDDEN,), 1),
        "language_model.mtp.pre_fc_norm_embedding.weight": _fixture((HIDDEN,), 2),
        "language_model.mtp.fc.weight": _fixture((HIDDEN, 2 * HIDDEN), 3),
        "language_model.mtp.norm.weight": _fixture((HIDDEN,), 4),
    }
    for layer_idx in range(mtp_layers):
        prefix = f"language_model.mtp.layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": _fixture((HIDDEN,), 10),
                f"{prefix}.post_attention_layernorm.weight": _fixture((HIDDEN,), 11),
                f"{prefix}.self_attn.q_proj.weight": _fixture(
                    (2 * NUM_HEADS * HEAD_DIM, HIDDEN), 12
                ),
                f"{prefix}.self_attn.k_proj.weight": _fixture(
                    (NUM_KV_HEADS * HEAD_DIM, HIDDEN), 13
                ),
                f"{prefix}.self_attn.v_proj.weight": _fixture(
                    (NUM_KV_HEADS * HEAD_DIM, HIDDEN), 14
                ),
                f"{prefix}.self_attn.o_proj.weight": _fixture(
                    (HIDDEN, NUM_HEADS * HEAD_DIM), 15
                ),
                f"{prefix}.self_attn.q_norm.weight": _fixture((HEAD_DIM,), 16),
                f"{prefix}.self_attn.k_norm.weight": _fixture((HEAD_DIM,), 17),
            }
        )
        mlp = f"{prefix}.mlp"
        if moe:
            weights.update(
                {
                    f"{mlp}.gate.weight": _fixture((NUM_EXPERTS, HIDDEN), 20),
                    f"{mlp}.switch_mlp.gate_proj.weight": _fixture(
                        (NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN), 21
                    ),
                    f"{mlp}.switch_mlp.up_proj.weight": _fixture(
                        (NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN), 22
                    ),
                    f"{mlp}.switch_mlp.down_proj.weight": _fixture(
                        (NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE), 23
                    ),
                    f"{mlp}.shared_expert.gate_proj.weight": _fixture(
                        (SHARED_INTERMEDIATE, HIDDEN), 24
                    ),
                    f"{mlp}.shared_expert.up_proj.weight": _fixture(
                        (SHARED_INTERMEDIATE, HIDDEN), 25
                    ),
                    f"{mlp}.shared_expert.down_proj.weight": _fixture(
                        (HIDDEN, SHARED_INTERMEDIATE), 26
                    ),
                    f"{mlp}.shared_expert_gate.weight": _fixture((1, HIDDEN), 27),
                }
            )
        else:
            weights.update(
                {
                    f"{mlp}.gate_proj.weight": _fixture((INTERMEDIATE, HIDDEN), 20),
                    f"{mlp}.up_proj.weight": _fixture((INTERMEDIATE, HIDDEN), 21),
                    f"{mlp}.down_proj.weight": _fixture((HIDDEN, INTERMEDIATE), 22),
                }
            )
    return weights


def _checkpoint(model, *, moe=False):
    backbone = {
        key: value
        for key, value in tree_flatten(model.parameters())
        if not key.startswith("language_model.mtp.")
    }
    backbone.update(
        _canonical_mtp_weights(
            moe=moe, mtp_layers=model.language_model.args.mtp_num_hidden_layers
        )
    )
    return backbone


def _sanitize_and_load(model, weights):
    sanitized = model.sanitize(weights)
    assert model.mtp_capability.reason == "native_mtp_weights_not_loaded"
    model.load_weights(list(sanitized.items()), strict=True)
    assert model.supports_mtp
    return sanitized


def _convert_moe_layout(weights, layout):
    prefix = "language_model.mtp.layers.0.mlp"
    gate = weights.pop(f"{prefix}.switch_mlp.gate_proj.weight")
    up = weights.pop(f"{prefix}.switch_mlp.up_proj.weight")
    down = weights.pop(f"{prefix}.switch_mlp.down_proj.weight")
    if layout == "fused":
        weights[f"{prefix}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=-2)
        weights[f"{prefix}.experts.down_proj"] = down
    elif layout == "per_expert":
        for expert in range(NUM_EXPERTS):
            weights[f"{prefix}.experts.{expert}.gate_proj.weight"] = gate[expert]
            weights[f"{prefix}.experts.{expert}.up_proj.weight"] = up[expert]
            weights[f"{prefix}.experts.{expert}.down_proj.weight"] = down[expert]
    else:
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = up
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = down


def _quantizable_prefixes(*, moe=False):
    layer = "language_model.mtp.layers.0"
    prefixes = {
        "language_model.mtp.fc",
        *(
            f"{layer}.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
    }
    mlp = f"{layer}.mlp"
    if moe:
        prefixes.update(
            {
                f"{mlp}.gate",
                f"{mlp}.shared_expert_gate",
                *(
                    f"{mlp}.switch_mlp.{name}"
                    for name in ("gate_proj", "up_proj", "down_proj")
                ),
                *(
                    f"{mlp}.shared_expert.{name}"
                    for name in ("gate_proj", "up_proj", "down_proj")
                ),
            }
        )
    else:
        prefixes.update(
            f"{mlp}.{name}" for name in ("gate_proj", "up_proj", "down_proj")
        )
    return prefixes


def test_config_and_missing_weights_do_not_publish_capability():
    model = _model()
    assert model.mtp_capability.reason == "native_mtp_weights_not_validated"
    with pytest.raises(ValueError, match="Native Qwen MTP weights"):
        model.sanitize({})
    assert not model.supports_mtp


def test_no_head_model_preserves_normal_forward_and_refuses_dispatch():
    model = _model(mtp_layers=0)
    logits = model(mx.array([[1, 2, 3]], dtype=mx.uint32))
    mx.eval(logits)
    assert logits.shape == (1, 3, 64)
    assert model.mtp_capability.reason == "native_mtp_head_not_configured"
    with pytest.raises(RuntimeError, match="native_mtp_head_not_configured"):
        model.make_mtp_cache()


def test_exact_sanitize_load_handshake_rejects_filtered_non_strict_load():
    model = _model()
    sanitized = model.sanitize(_checkpoint(model))
    omitted = "language_model.mtp.fc.weight"
    filtered = [(key, value) for key, value in sanitized.items() if key != omitted]
    model.load_weights(filtered, strict=False)
    assert model.mtp_capability.reason == "native_mtp_weights_not_loaded"

    # The failed handshake is one-shot: replaying the old mapping cannot enable MTP.
    model.load_weights(list(sanitized.items()), strict=True)
    assert model.mtp_capability.reason == "native_mtp_weights_not_loaded"

    _sanitize_and_load(model, _checkpoint(model))


def test_failed_resanitize_invalidates_an_older_pending_handshake():
    model = _model()
    previously_sanitized = model.sanitize(_checkpoint(model))
    invalid = _checkpoint(model)
    invalid.pop("language_model.mtp.fc.weight")
    with pytest.raises(ValueError, match="missing language_model.mtp.fc.weight"):
        model.sanitize(invalid)

    model.load_weights(list(previously_sanitized.items()), strict=True)
    assert model.mtp_capability.reason == "native_mtp_weights_not_validated"


def test_exact_strict_false_wrong_shape_load_is_rejected_and_not_activated():
    model = _model()
    weights = _checkpoint(model)
    weights["language_model.mtp.fc.weight"] = _fixture((HIDDEN - 1, 2 * HIDDEN), 101)
    sanitized = model.sanitize(weights)
    with pytest.raises(ValueError, match="MTP load leaf type/shape mismatch"):
        model.load_weights(list(sanitized.items()), strict=False)
    assert model.mtp_capability.reason == "native_mtp_weights_not_loaded"


def test_dense_mtp_preserves_normal_logits_and_returns_numeric_pre_norm_hidden():
    model = _model(tie_word_embeddings=False)
    inputs = mx.array([[1, 2, 3]], dtype=mx.uint32)
    baseline = model(inputs)
    _sanitize_and_load(model, _checkpoint(model))
    logits, hidden = model(inputs, return_hidden=True)
    normed = model.model.norm(hidden)
    expected = model.language_model.lm_head(normed)
    incorrectly_post_norm_contract = model.language_model.lm_head(hidden)
    mx.eval(baseline, logits, hidden, normed, expected, incorrectly_post_norm_contract)
    assert mx.allclose(baseline, logits).item()
    assert mx.allclose(logits, expected).item()
    assert not mx.allclose(logits, incorrectly_post_norm_contract, atol=1e-5).item()

    mtp_cache = model.make_mtp_cache()
    mtp_hidden = model.language_model.mtp(
        hidden[:, -1:], mx.array([[4]]), model.model.embed_tokens, mtp_cache
    )
    shared_head_logits = model.language_model.lm_head(mtp_hidden)
    actual = model.mtp_forward(hidden[:, -1:], mx.array([[4]]), model.make_mtp_cache())
    mx.eval(shared_head_logits, actual)
    assert mx.allclose(actual, shared_head_logits).item()


def test_vlm_wrapper_mtp_keys_map_beside_backbone():
    model = _model()
    weights = _checkpoint(model)
    for key in list(weights):
        if key.startswith("language_model.mtp."):
            suffix = key.removeprefix("language_model.")
            weights[f"model.language_model.{suffix}"] = weights.pop(key)
    sanitized = _sanitize_and_load(model, weights)
    assert not any(key.startswith("language_model.model.mtp.") for key in sanitized)


@pytest.mark.parametrize("layout", ["fused", "per_expert", "stacked"])
def test_moe_accepts_independent_canonical_expert_layouts(layout):
    model = _model(moe=True)
    weights = _checkpoint(model, moe=True)
    _convert_moe_layout(weights, layout)
    sanitized = _sanitize_and_load(model, weights)
    prefix = "language_model.mtp.layers.0.mlp.switch_mlp"
    assert all(
        f"{prefix}.{name}.weight" in sanitized
        for name in ("gate_proj", "up_proj", "down_proj")
    )


def test_moe_rejects_mixed_expert_layouts():
    model = _model(moe=True)
    weights = _checkpoint(model, moe=True)
    weights["language_model.mtp.layers.0.mlp.experts.gate_up_proj"] = _fixture(
        (NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN), 99
    )
    with pytest.raises(ValueError, match="Mixed MoE expert layouts"):
        model.sanitize(weights)


@pytest.mark.parametrize("moe", [False, True])
def test_quantized_mtp_complete_triplets_load_but_partial_triplets_fail(moe):
    model = _model(moe=moe)
    weights = _checkpoint(model, moe=moe)
    prefixes = _quantizable_prefixes(moe=moe)
    for prefix in prefixes:
        weight_key = f"{prefix}.weight"
        quantized, scales, biases = mx.quantize(
            weights[weight_key], group_size=32, bits=4
        )
        weights[weight_key] = quantized
        weights[f"{prefix}.scales"] = scales
        weights[f"{prefix}.biases"] = biases

    sanitized = model.sanitize(weights)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            path in prefixes and hasattr(module, "to_quantized")
        ),
    )
    # Exact, shape-valid post-quantization mappings may activate under
    # strict=False; the safety decision is independent of unrelated leaves.
    model.load_weights(list(sanitized.items()), strict=False)
    assert model.supports_mtp

    partial_model = _model(moe=moe)
    partial = _checkpoint(partial_model, moe=moe)
    prefix = "language_model.mtp.fc"
    _, scales, _ = mx.quantize(partial[f"{prefix}.weight"], group_size=32, bits=4)
    partial[f"{prefix}.scales"] = scales
    with pytest.raises(ValueError, match="incomplete quantized triplet"):
        partial_model.sanitize(partial)


def test_tensor_sharding_fails_before_group_or_parameter_mutation():
    model = _model()
    before = [(key, id(value)) for key, value in tree_flatten(model.parameters())]

    class ExplodingGroup:
        def rank(self):
            raise AssertionError("group inspected before MTP sharding guard")

        def size(self):
            raise AssertionError("group inspected before MTP sharding guard")

    with pytest.raises(RuntimeError, match="does not support tensor/distributed"):
        model.shard(ExplodingGroup())
    after = [(key, id(value)) for key, value in tree_flatten(model.parameters())]
    assert before == after


def test_pipeline_parallelism_remains_ineligible():
    model = _model()
    _sanitize_and_load(model, _checkpoint(model))

    class TwoRanks:
        def rank(self):
            return 0

        def size(self):
            return 2

    model.model.pipeline(TwoRanks())
    assert model.mtp_capability.reason == "native_mtp_pipeline_parallelism_unsupported"
    with pytest.raises(RuntimeError, match="pipeline_parallelism_unsupported"):
        model.make_mtp_cache()
