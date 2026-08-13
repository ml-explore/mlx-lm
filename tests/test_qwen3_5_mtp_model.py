# Copyright © 2026 Apple Inc.
"""Synthetic model-only coverage for Qwen native MTP capability handling."""

import importlib
import math
from contextlib import contextmanager
from contextvars import ContextVar

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_lm.generate import (
    GenerationForward,
    GenerationForwardPhase,
    GenerationForwardPositionAck,
    NativeMTPSparseBootstrap,
    attested_target_forward,
    mtp_generate_step,
    stream_generate,
)
from mlx_lm.models.cache import ArraysCache, KVCache, NativeMTPRequestCache
from mlx_lm.models.generation_context import (
    consume_generation_positions,
    generation_forward_context,
)

HIDDEN = 32
INTERMEDIATE = 64
HEAD_DIM = 8
NUM_HEADS = 4
NUM_KV_HEADS = 2
NUM_EXPERTS = 2
MOE_INTERMEDIATE = 32
SHARED_INTERMEDIATE = 64
_ACTIVE_FORWARD = ContextVar("qwen3_5_mtp_wrapper_forward", default=None)


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


def _model(*, moe=False, mtp_layers=1, tie_word_embeddings=True, acknowledge=False):
    module_name = "mlx_lm.models.qwen3_5_moe" if moe else "mlx_lm.models.qwen3_5"
    module = importlib.import_module(module_name)
    model_type = module.Model
    if acknowledge:

        class _AcknowledgingModel(model_type):
            @staticmethod
            def _acknowledge_positions():
                forward = _ACTIVE_FORWARD.get()
                if forward is not None and forward.logical_position_ack is not None:
                    forward.logical_position_ack.acknowledge(forward.logical_positions)

            def __call__(self, *args, **kwargs):
                self._acknowledge_positions()
                return super().__call__(*args, **kwargs)

            def mtp_forward(self, *args, **kwargs):
                self._acknowledge_positions()
                return super().mtp_forward(*args, **kwargs)

        model_type = _AcknowledgingModel
    model = model_type(
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


def _positioned_target(model, inputs, cache, positions):
    ack = GenerationForwardPositionAck(
        positions,
        model=model,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
    )
    forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=positions,
        logical_position_ack=ack,
    )
    with generation_forward_context(forward):
        ack._activate()
        try:
            result = model(inputs, cache=cache, return_hidden=True)
            ack._require_acknowledged()
            return result
        finally:
            ack._finish()


def _positioned_mtp(model, hidden, next_tokens, cache, positions):
    ack = GenerationForwardPositionAck(
        positions,
        model=model,
        cache=cache,
        phase=GenerationForwardPhase.MTP_DRAFT,
    )
    forward = GenerationForward(
        model=model,
        input_tokens=next_tokens,
        cache=cache,
        phase=GenerationForwardPhase.MTP_DRAFT,
        logical_positions=positions,
        logical_position_ack=ack,
    )
    with generation_forward_context(forward):
        ack._activate()
        try:
            result = model.mtp_forward(hidden, next_tokens, cache)
            ack._require_acknowledged()
            return result
        finally:
            ack._finish()


def _assert_float_payload_matches(actual, expected):
    # Batched and B=1 matmuls may differ in accumulation order by ~1e-6.
    assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item()


def _assert_batched_cache_matches_independent_rows(batched, independent_rows):
    """Compare active cache payloads and recurrent metadata, row by row."""

    assert len(batched) == len(independent_rows[0])
    for entry_index, batched_entry in enumerate(batched):
        row_entries = [row[entry_index] for row in independent_rows]
        assert all(type(row_entry) is type(batched_entry) for row_entry in row_entries)
        if isinstance(batched_entry, ArraysCache):
            for state_index, value in enumerate(batched_entry.cache):
                if value is None:
                    assert all(row.cache[state_index] is None for row in row_entries)
                else:
                    for row_index, row in enumerate(row_entries):
                        _assert_float_payload_matches(
                            value[row_index : row_index + 1], row.cache[state_index]
                        )
            for name in ("left_padding", "lengths"):
                value = getattr(batched_entry, name)
                if value is None:
                    assert all(getattr(row, name) is None for row in row_entries)
                else:
                    for row_index, row in enumerate(row_entries):
                        assert mx.array_equal(
                            value[row_index : row_index + 1], getattr(row, name)
                        ).item()
            continue
        assert isinstance(batched_entry, KVCache)
        assert all(row.offset == batched_entry.offset for row in row_entries)
        for name in ("keys", "values"):
            value = getattr(batched_entry, name)
            for row_index, row in enumerate(row_entries):
                _assert_float_payload_matches(
                    value[row_index : row_index + 1], getattr(row, name)
                )


@contextmanager
def _acknowledging_forward_context(forward):
    token = _ACTIVE_FORWARD.set(forward)
    try:
        yield
    finally:
        _ACTIVE_FORWARD.reset(token)


class _StreamDetokenizer:
    def __init__(self):
        self.last_segment = ""
        self.finalized = False

    def add_token(self, token):
        self.last_segment = str(token)

    def finalize(self):
        self.finalized = True


class _StreamTokenizer:
    def __init__(self):
        self.detokenizer = _StreamDetokenizer()
        self.eos_token_ids = frozenset()
        self.eos_token_id = 0
        self.chat_template = None

    @staticmethod
    def get_vocab():
        return {}

    @staticmethod
    def encode(_text, *, add_special_tokens=False):
        return [1]

    @staticmethod
    def decode(tokens):
        return "".join(str(token) for token in tokens)


def _sparse_bootstrap(model):
    target_cache = model.make_cache()
    (_, _), receipt = attested_target_forward(
        model,
        (0, 1),
        target_cache,
        phase=GenerationForwardPhase.PREFILL,
        logical_positions=(0, 1),
        immediate_successor_token_ids=(1,),
        model_forward_context=model.generation_forward_context,
    )
    return NativeMTPSparseBootstrap(
        receipts=(receipt,),
        selected_logical_positions=(0, 1),
        selected_token_ids=(0, 1),
        immediate_successor_token_ids=(1,),
        target_cache=target_cache,
        next_logical_position=2,
    )


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


@pytest.mark.parametrize("moe", (False, True))
def test_dense_and_moe_models_expose_the_same_structured_mtp_cache_contract(moe):
    model = _model(moe=moe)
    _sanitize_and_load(model, _checkpoint(model, moe=moe))

    request = model.make_mtp_request_cache()
    assert model.mtp is model.language_model.mtp
    assert request.model is model.language_model
    assert not any(
        key.startswith("mtp.") for key, _ in tree_flatten(model.parameters())
    )
    assert request.backbone is not request.mtp
    assert request.state.backbone_tokens == 0
    assert request.state.mtp_tokens == 0
    request.assert_aligned(backbone_tokens=0, mtp_tokens=0)
    with pytest.raises(ValueError, match="native_mtp_prefix_reuse_unsupported"):
        model.make_mtp_request_cache(prompt_cache=model.make_cache())


def test_outer_mtp_topology_view_preserves_unloaded_fail_closed_behavior():
    model = _model()
    target_cache = model.make_cache()

    assert model.mtp is model.language_model.mtp
    with pytest.raises(RuntimeError, match="native_mtp_weights_not_validated"):
        NativeMTPRequestCache.adopt_sparse_target(
            model,
            target_cache=target_cache,
            target_tokens=1,
            next_logical_position=1,
        )

    assert target_cache[1].offset == 0


@pytest.mark.parametrize("moe", (False, True))
def test_outer_dense_and_moe_sparse_bootstrap_stream_adopts_outer_topology(
    monkeypatch, moe
):
    model = _model(moe=moe, acknowledge=True)
    _sanitize_and_load(model, _checkpoint(model, moe=moe))
    bootstrap = _sparse_bootstrap(model)
    adopted = []
    sealed = []
    original_adopt = NativeMTPRequestCache.adopt_sparse_target.__func__
    original_seal = NativeMTPRequestCache.seal_verified

    def capture_adopt(cls, *args, **kwargs):
        request = original_adopt(cls, *args, **kwargs)
        adopted.append(request)
        return request

    def capture_seal(self, *, backbone_tokens, mtp_tokens):
        result = original_seal(
            self, backbone_tokens=backbone_tokens, mtp_tokens=mtp_tokens
        )
        sealed.append((backbone_tokens, mtp_tokens, result))
        return result

    generate_module = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    monkeypatch.setattr(
        NativeMTPRequestCache, "adopt_sparse_target", classmethod(capture_adopt)
    )
    monkeypatch.setattr(NativeMTPRequestCache, "seal_verified", capture_seal)
    tokenizer = _StreamTokenizer()

    responses = list(
        stream_generate(
            model,
            tokenizer,
            None,
            mtp=True,
            max_tokens=1,
            sparse_bootstrap=bootstrap,
        )
    )

    request = adopted[0]
    assert request.model is model
    assert request.backbone is bootstrap.target_cache
    assert (sealed[0][0], sealed[0][1]) == (2, 1)
    assert sealed[0][2].backbone_tokens == 2
    assert sealed[0][2].mtp_tokens == 1
    assert request.closed
    assert not request.checkpoint_active
    assert tokenizer.detokenizer.finalized
    assert responses[-1].prompt_tokens == 2
    assert responses[-1].generation_tokens == 1


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


@pytest.mark.parametrize("moe", [False, True])
def test_batched_position_matrix_matches_independent_rows_for_target_and_mtp(moe):
    """A real B=2 graph honours unequal, non-contiguous row positions."""

    model = _model(moe=moe)
    _sanitize_and_load(model, _checkpoint(model, moe=moe))
    inputs = mx.array(((3, 7), (11, 5)), dtype=mx.uint32)
    positions = ((5, 19), (103, 701))

    batched_cache = model.make_cache()
    batched_logits, batched_hidden = _positioned_target(
        model, inputs, batched_cache, positions
    )
    assert batched_cache[-1].offset == inputs.shape[1]

    single_targets = []
    single_hidden = []
    single_backbone_caches = []
    for row, row_positions in zip(inputs, positions):
        row_cache = model.make_cache()
        logits, hidden = _positioned_target(model, row[None], row_cache, row_positions)
        single_targets.append(logits)
        single_hidden.append(hidden)
        single_backbone_caches.append(row_cache)
        assert row_cache[-1].offset == inputs.shape[1]
    _assert_float_payload_matches(
        batched_logits, mx.concatenate(single_targets, axis=0)
    )
    _assert_float_payload_matches(batched_hidden, mx.concatenate(single_hidden, axis=0))
    _assert_batched_cache_matches_independent_rows(
        batched_cache, single_backbone_caches
    )

    next_tokens = mx.array(((17, 23), (29, 31)), dtype=mx.uint32)
    batched_mtp_cache = model.make_mtp_cache()
    batched_mtp = _positioned_mtp(
        model, batched_hidden, next_tokens, batched_mtp_cache, positions
    )
    assert batched_mtp_cache[-1].offset == next_tokens.shape[1]

    single_mtp = []
    single_mtp_caches = []
    for index, row_positions in enumerate(positions):
        row_mtp_cache = model.make_mtp_cache()
        logits = _positioned_mtp(
            model,
            single_hidden[index],
            next_tokens[index : index + 1],
            row_mtp_cache,
            row_positions,
        )
        single_mtp.append(logits)
        single_mtp_caches.append(row_mtp_cache)
        assert row_mtp_cache[-1].offset == next_tokens.shape[1]
    _assert_float_payload_matches(batched_mtp, mx.concatenate(single_mtp, axis=0))
    _assert_batched_cache_matches_independent_rows(batched_mtp_cache, single_mtp_caches)

    # A later decode must retain both numerical outputs and every active cache
    # payload after the heterogeneous first-position matrix.
    continuation = mx.array(((37,), (43,)), dtype=mx.uint32)
    continuation_positions = ((1003,), (9007,))
    continued_logits, continued_hidden = _positioned_target(
        model, continuation, batched_cache, continuation_positions
    )
    row_continued_logits = []
    row_continued_hidden = []
    for index, row_positions in enumerate(continuation_positions):
        logits, hidden = _positioned_target(
            model,
            continuation[index : index + 1],
            single_backbone_caches[index],
            row_positions,
        )
        row_continued_logits.append(logits)
        row_continued_hidden.append(hidden)
    _assert_float_payload_matches(
        continued_logits, mx.concatenate(row_continued_logits, axis=0)
    )
    _assert_float_payload_matches(
        continued_hidden, mx.concatenate(row_continued_hidden, axis=0)
    )
    assert batched_cache[-1].offset == 3
    _assert_batched_cache_matches_independent_rows(
        batched_cache, single_backbone_caches
    )

    continued_mtp_tokens = mx.array(((47,), (53,)), dtype=mx.uint32)
    continued_mtp = _positioned_mtp(
        model,
        continued_hidden,
        continued_mtp_tokens,
        batched_mtp_cache,
        continuation_positions,
    )
    row_continued_mtp = []
    for index, row_positions in enumerate(continuation_positions):
        row_continued_mtp.append(
            _positioned_mtp(
                model,
                row_continued_hidden[index],
                continued_mtp_tokens[index : index + 1],
                single_mtp_caches[index],
                row_positions,
            )
        )
    _assert_float_payload_matches(
        continued_mtp, mx.concatenate(row_continued_mtp, axis=0)
    )
    assert batched_mtp_cache[-1].offset == 3
    _assert_batched_cache_matches_independent_rows(batched_mtp_cache, single_mtp_caches)


@pytest.mark.parametrize("moe", [False, True])
def test_positioned_qwen_target_and_mtp_are_sensitive_to_each_row_position(moe):
    """Prevent a B=2 position-matrix acknowledgement from becoming a no-op."""

    model = _model(moe=moe)
    _sanitize_and_load(model, _checkpoint(model, moe=moe))
    inputs = mx.array(((3, 7), (11, 5)), dtype=mx.uint32)
    heterogeneous = ((5, 19), (103, 701))
    shifted = ((17, 47), (127, 761))

    first_target_cache = model.make_cache()
    first_logits, first_hidden = _positioned_target(
        model, inputs, first_target_cache, heterogeneous
    )
    second_target_cache = model.make_cache()
    second_logits, second_hidden = _positioned_target(
        model, inputs, second_target_cache, shifted
    )

    # The tiny real-Qwen fixture changes logits by about 1e-6.  Active KV keys
    # are the direct positional-consumption oracle; exact output inequality
    # still proves the position matrix reaches the target calculation.
    assert not mx.array_equal(first_logits, second_logits).item()
    assert not mx.array_equal(first_hidden, second_hidden).item()
    assert not mx.array_equal(
        first_target_cache[-1].keys, second_target_cache[-1].keys
    ).item()

    next_tokens = mx.array(((17, 23), (29, 31)), dtype=mx.uint32)
    first_mtp_cache = model.make_mtp_cache()
    first_mtp_logits = _positioned_mtp(
        model, first_hidden, next_tokens, first_mtp_cache, heterogeneous
    )
    second_mtp_cache = model.make_mtp_cache()
    second_mtp_logits = _positioned_mtp(
        model, first_hidden, next_tokens, second_mtp_cache, shifted
    )

    # Qwen's native MTP topology is attention-only, so its active keys are an
    # independent direct oracle that the MTP context consumed the matrix too.
    assert not mx.array_equal(first_mtp_logits, second_mtp_logits).item()
    assert not mx.array_equal(
        first_mtp_cache[-1].keys, second_mtp_cache[-1].keys
    ).item()


@pytest.mark.parametrize(
    ("positions", "message"),
    (
        ((1, 2), "batch_requires_matrix"),
        (((1, 2), (3,)), "matrix_ragged"),
        (((1, 2, 3), (4, 5, 6)), "input_shape_mismatch"),
        (((1, -2), (3, 4)), "value_invalid"),
        (((1, True), (3, 4)), "value_invalid"),
        (((1, "two"), (3, 4)), "value_invalid"),
    ),
)
def test_batched_position_matrix_rejects_broadcast_ragged_and_malformed_inputs(
    positions, message
):
    with pytest.raises(ValueError, match=message):
        GenerationForward(
            model=object(),
            input_tokens=mx.zeros((2, 2), dtype=mx.uint32),
            cache=[],
            phase=GenerationForwardPhase.VERIFY,
            logical_positions=positions,
        )


def test_position_ack_is_one_shot_and_context_resets_after_model_error():
    model = _model()
    _sanitize_and_load(model, _checkpoint(model))
    inputs = mx.array(((3,),), dtype=mx.uint32)
    cache = model.make_cache()
    ack = GenerationForwardPositionAck(
        (37,), model=model, cache=cache, phase=GenerationForwardPhase.VERIFY
    )
    forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=(37,),
        logical_position_ack=ack,
    )
    with generation_forward_context(forward):
        ack._activate()
        try:
            model(inputs, cache=cache, return_hidden=True)
            with pytest.raises(RuntimeError, match="ack_reused"):
                ack.acknowledge((37,))
        finally:
            ack._finish()

    # A failing scoped call must not leak positions into the next unscoped call.
    with pytest.raises(RuntimeError, match="forced position-context failure"):
        with generation_forward_context(forward):
            raise RuntimeError("forced position-context failure")
    fresh_cache = model.make_cache()
    logits = model(inputs, cache=fresh_cache)
    assert logits.shape[:2] == (1, 1)


def test_single_row_rejects_matrix_form_before_the_model_is_called():
    with pytest.raises(ValueError, match="single_row_requires_vector"):
        GenerationForward(
            model=object(),
            input_tokens=mx.zeros((1, 2), dtype=mx.uint32),
            cache=[],
            phase=GenerationForwardPhase.VERIFY,
            logical_positions=((1, 2),),
        )


def test_position_context_rejects_model_cache_phase_and_capability_mismatches():
    model = _model()
    other_model = _model()
    _sanitize_and_load(model, _checkpoint(model))
    _sanitize_and_load(other_model, _checkpoint(other_model))
    inputs = mx.array(((3,),), dtype=mx.uint32)
    cache = model.make_cache()
    ack = GenerationForwardPositionAck(
        (11,), model=model, cache=cache, phase=GenerationForwardPhase.VERIFY
    )
    forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=(11,),
        logical_position_ack=ack,
    )

    with generation_forward_context(forward):
        ack._activate()
        try:
            with pytest.raises(RuntimeError, match="model_mismatch"):
                other_model(inputs, cache=cache, return_hidden=True)
        finally:
            ack._finish()

    wrong_cache = model.make_cache()
    ack = GenerationForwardPositionAck(
        (11,), model=model, cache=cache, phase=GenerationForwardPhase.VERIFY
    )
    forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=(11,),
        logical_position_ack=ack,
    )
    with generation_forward_context(forward):
        ack._activate()
        try:
            with pytest.raises(RuntimeError, match="cache_mismatch"):
                model(inputs, cache=wrong_cache, return_hidden=True)
        finally:
            ack._finish()

    mtp_cache = model.make_mtp_cache()
    phase_ack = GenerationForwardPositionAck(
        (11,), model=model, cache=mtp_cache, phase=GenerationForwardPhase.VERIFY
    )
    phase_forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=mtp_cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=(11,),
        logical_position_ack=phase_ack,
    )
    with generation_forward_context(phase_forward):
        phase_ack._activate()
        try:
            with pytest.raises(RuntimeError, match="phase_mismatch"):
                model.mtp_forward(mx.zeros((1, 1, HIDDEN)), inputs, mtp_cache)
        finally:
            phase_ack._finish()

    capability_ack = GenerationForwardPositionAck(
        (11,), model=model, cache=cache, phase=GenerationForwardPhase.VERIFY
    )
    capability_forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=(11,),
        logical_position_ack=capability_ack,
    )
    model.language_model._mtp_weights_loaded = False
    try:
        with generation_forward_context(capability_forward):
            capability_ack._activate()
            try:
                with pytest.raises(RuntimeError, match="capability_mismatch"):
                    model(inputs, cache=cache, return_hidden=True)
            finally:
                capability_ack._finish()
    finally:
        model.language_model._mtp_weights_loaded = True


def test_activated_model_error_resets_the_position_context():
    class ExplodingModel:
        generation_forward_context = staticmethod(generation_forward_context)

        def __call__(self, inputs, *, cache):
            consume_generation_positions(self, cache, inputs, mtp=False)
            raise RuntimeError("forced model failure")

    model = ExplodingModel()
    inputs = mx.array(((3,),), dtype=mx.uint32)
    cache = []
    ack = GenerationForwardPositionAck(
        (11,), model=model, cache=cache, phase=GenerationForwardPhase.VERIFY
    )
    forward = GenerationForward(
        model=model,
        input_tokens=inputs,
        cache=cache,
        phase=GenerationForwardPhase.VERIFY,
        logical_positions=(11,),
        logical_position_ack=ack,
    )
    with pytest.raises(RuntimeError, match="forced model failure"):
        with generation_forward_context(forward):
            ack._activate()
            try:
                model(inputs, cache=cache)
            finally:
                ack._finish()

    # The following unscoped call proves the failing active context was reset.
    with pytest.raises(RuntimeError, match="forced model failure"):
        model(inputs, cache=[])


def test_position_matrix_rope_path_has_no_host_row_extraction_or_sync():
    import inspect
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    source = inspect.getsource(Qwen3NextAttention.__call__)
    assert "B * L" in source
    assert ".item(" not in source
    assert ".tolist(" not in source
    assert "for row" not in source


def test_single_row_vector_is_compatible_with_the_existing_dense_call_shape():
    model = _model()
    _sanitize_and_load(model, _checkpoint(model))
    inputs = mx.array(((3, 7),), dtype=mx.uint32)

    ordinary_cache = model.make_cache()
    ordinary_logits, ordinary_hidden = model(
        inputs, cache=ordinary_cache, return_hidden=True
    )
    positioned_cache = model.make_cache()
    positioned_logits, positioned_hidden = _positioned_target(
        model, inputs, positioned_cache, (0, 1)
    )

    _assert_float_payload_matches(ordinary_logits, positioned_logits)
    _assert_float_payload_matches(ordinary_hidden, positioned_hidden)
    assert ordinary_cache[-1].offset == positioned_cache[-1].offset == 2


def test_positioned_mtp_generate_uses_qwen_context_without_manual_ack():
    model = _model()
    _sanitize_and_load(model, _checkpoint(model))
    telemetry = {}

    output = list(
        mtp_generate_step(
            mx.array((3, 7), dtype=mx.uint32),
            model,
            max_tokens=1,
            telemetry=telemetry,
            prompt_logical_positions=(41, 97),
        )
    )

    assert len(output) == 1
    assert telemetry["mtp_bypass_reason"] is None


def test_positioned_stream_generate_uses_qwen_context_without_callback(monkeypatch):
    model = _model()
    _sanitize_and_load(model, _checkpoint(model))
    generate_module = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(generate_module, "TokenizerWrapper", _StreamTokenizer)
    tokenizer = _StreamTokenizer()

    output = list(
        stream_generate(
            model,
            tokenizer,
            mx.array((3, 7), dtype=mx.uint32),
            mtp=True,
            max_tokens=1,
            prompt_logical_positions=(41, 97),
        )
    )

    assert output[-1].mtp_bypass_reason is None
    assert tokenizer.detokenizer.finalized


def test_positioned_qwen_rejects_manual_ack_context_as_ambiguous_before_forward():
    model = _model()
    _sanitize_and_load(model, _checkpoint(model))

    @contextmanager
    def manual_ack(forward):
        if forward.logical_position_ack is not None:
            forward.logical_position_ack.acknowledge(forward.logical_positions)
        yield

    with pytest.raises(ValueError, match="native_mtp_position_context_ambiguous"):
        list(
            mtp_generate_step(
                mx.array((3, 7), dtype=mx.uint32),
                model,
                max_tokens=1,
                prompt_logical_positions=(41, 97),
                model_forward_context=manual_ack,
            )
        )
