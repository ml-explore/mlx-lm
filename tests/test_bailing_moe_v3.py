from dataclasses import replace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_lm.models.bailing_moe_v3 import (
    Model,
    ModelArgs,
    _is_kda_layer,
    _normalize_kda_qk,
)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear


def _model_args(granularity):
    return ModelArgs(
        architectures=["BailingMoeV3ForCausalLM"],
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        n_group=1,
        topk_group=1,
        layer_group_size=1,
        head_dim=32,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        qk_head_dim=32,
        v_head_dim=32,
        gated_attention_proj_granularity_type=granularity,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )


def _bf16_model_args():
    return replace(
        _model_args(None),
        quantization_config=None,
        num_hidden_layers=4,
        layer_group_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )


def test_incomplete_layer_group_tail_uses_mla():
    assert [i for i in range(5) if _is_kda_layer(i, 2, 5)] == [0, 2]


def test_kda_qk_normalization_matches_reference_l2norm_epsilon():
    head_dim = 32
    q = mx.arange(2 * head_dim, dtype=mx.float32).reshape(1, 2, 1, head_dim)
    k = mx.arange(1, 2 * head_dim + 1, dtype=mx.float32).reshape(1, 2, 1, head_dim)
    q = (q - head_dim) * 1e-4
    k = (k - head_dim) * 1e-4

    actual_q, actual_k = _normalize_kda_qk(q, k, head_dim)
    ref_q = q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6)
    ref_k = k * mx.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)
    ref_q = (head_dim**-0.5) * ref_q

    assert mx.allclose(actual_q, ref_q, rtol=1e-5, atol=1e-7)
    assert mx.allclose(actual_k, ref_k, rtol=1e-5, atol=1e-7)


def test_rejects_non_fp8_checkpoint_config():
    with pytest.raises(ValueError, match="requires an FP8 E4M3 checkpoint"):
        Model(ModelArgs(quantization_config={"quant_method": "fp8"}))


def test_bf16_model_uses_unquantized_projection_layers():
    model = Model(_bf16_model_args())

    assert isinstance(model.layers[0].attention.q_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.gate_proj, nn.Linear)
    assert isinstance(model.layers[1].mlp.switch_mlp.gate_proj, SwitchLinear)
    assert isinstance(model.layers[3].attention.dense, nn.Linear)


def test_fp8_model_keeps_quantized_projection_layers():
    fp8_config = _model_args(None).quantization_config
    model = Model(replace(_bf16_model_args(), quantization_config=fp8_config))

    assert isinstance(model.layers[0].attention.q_proj, nn.QuantizedLinear)
    assert isinstance(model.layers[3].attention.dense, nn.QuantizedLinear)
    assert isinstance(model.layers[1].mlp.switch_mlp.gate_proj, QuantizedSwitchLinear)


def test_bf16_sanitize_stacks_expert_weights_without_scales():
    args = _bf16_model_args()
    model = Model(args)
    weights = {}
    projection_shapes = {
        "gate_proj": (args.moe_intermediate_size, args.hidden_size),
        "up_proj": (args.moe_intermediate_size, args.hidden_size),
        "down_proj": (args.hidden_size, args.moe_intermediate_size),
    }
    for layer_index in range(args.first_k_dense_replace, args.num_hidden_layers):
        for expert in range(args.num_experts):
            for projection, shape in projection_shapes.items():
                key = (
                    f"model.layers.{layer_index}.mlp.experts.{expert}."
                    f"{projection}.weight"
                )
                weights[key] = mx.full(shape, expert + 1, dtype=mx.bfloat16)

    weights = model.sanitize(weights)

    for layer_index in range(args.first_k_dense_replace, args.num_hidden_layers):
        for projection, shape in projection_shapes.items():
            key = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}.weight"
            assert weights[key].shape == (args.num_experts, *shape)
    assert not any(key.endswith(".scales") for key in weights)


def test_bf16_four_layer_group_forward():
    args = _bf16_model_args()
    checkpoint_model = Model(args)
    checkpoint_model.set_dtype(mx.bfloat16)

    checkpoint = {}
    for key, value in tree_flatten(checkpoint_model.parameters()):
        if ".switch_mlp." not in key or not key.endswith(".weight"):
            checkpoint[key] = value
            continue
        prefix, suffix = key.split(".switch_mlp.", 1)
        projection = suffix.removesuffix(".weight")
        for expert in range(args.num_experts):
            expert_key = f"{prefix}.experts.{expert}.{projection}.weight"
            checkpoint[expert_key] = value[expert]

    model = Model(args)
    weights = model.sanitize(checkpoint)
    model.load_weights(list(weights.items()), strict=True)

    logits = model(mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(logits)

    assert logits.shape == (1, 2, args.vocab_size)
    assert logits.dtype == mx.bfloat16


@pytest.mark.parametrize(
    "granularity",
    [None, "head_wise", "element_wise"],
)
def test_accepts_supported_mla_gate_granularities(granularity):
    model = Model(_model_args(granularity))
    attention = model.layers[0].attention
    if granularity is None:
        assert attention.g_proj is None
    else:
        assert attention.g_proj is not None


def test_rejects_unknown_mla_gate_granularity():
    with pytest.raises(ValueError, match="Unsupported gated_attention"):
        Model(_model_args("token_wise"))
