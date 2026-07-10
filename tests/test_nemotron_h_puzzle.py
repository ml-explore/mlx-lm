import json
from dataclasses import asdict, replace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import nemotron_h
from mlx_lm.models.cache import ArraysCache
from mlx_lm.utils import _get_classes, load_model, quantize_model, save_model


def puzzle_args():
    return nemotron_h.ModelArgs(
        model_type="nemotron_h_puzzle",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        max_position_embeddings=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=3,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        layers_block_type=["moe", "moe"],
        block_configs=[
            {
                "block_type": "moe",
                "moe_intermediate_size": 32,
                "num_experts_per_tok": 1,
            },
            {
                "block_type": "moe",
                "moe_intermediate_size": 64,
                "num_experts_per_tok": 2,
            },
        ],
        moe_shared_expert_intermediate_size=32,
        moe_latent_size=32,
        n_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def test_puzzle_model_type_remaps_to_nemotron_h():
    model_cls, args_cls = _get_classes({"model_type": "nemotron_h_puzzle"})
    assert model_cls is nemotron_h.Model
    assert args_cls is nemotron_h.ModelArgs


def test_puzzle_quantization_preserves_output_head():
    args = puzzle_args()
    model = nemotron_h.Model(args)

    quantize_model(model, asdict(args), group_size=32, bits=4)

    assert isinstance(model.lm_head, nn.Linear)
    assert not hasattr(model.lm_head, "scales")
    assert hasattr(model.layers[0].mixer.switch_mlp.fc1, "scales")


def test_official_config_derives_layer_count():
    config = asdict(puzzle_args())
    config.pop("num_hidden_layers")

    args = nemotron_h.ModelArgs.from_dict(config)

    assert args.num_hidden_layers == len(config["block_configs"])


def test_puzzle_layers_use_heterogeneous_moe_dimensions_and_top_k():
    model = nemotron_h.Model(puzzle_args())

    first, second = model.layers
    assert first.mixer.switch_mlp.fc1.output_dims == 32
    assert second.mixer.switch_mlp.fc1.output_dims == 64
    assert first.mixer.gate.top_k == 1
    assert second.mixer.gate.top_k == 2

    logits = model(mx.array([[1, 2, 3]]))
    mx.eval(logits)
    assert logits.shape == (1, 3, 128)


def test_mamba_prefill_matches_cached_decode():
    args = replace(
        puzzle_args(),
        num_hidden_layers=1,
        hybrid_override_pattern=["M"],
        layers_block_type=["mamba"],
        block_configs=[{"block_type": "mamba"}],
    )
    block = nemotron_h.NemotronHBlock(args.for_layer(0), "M")
    mx.random.seed(7)
    inputs = mx.random.normal((1, 4, args.hidden_size)).astype(mx.bfloat16)

    prefill = block(inputs)
    cache = ArraysCache(size=2)
    decoded = mx.concatenate(
        [
            block(inputs[:, position : position + 1], cache=cache)
            for position in range(4)
        ],
        axis=1,
    )
    mx.eval(prefill, decoded)

    assert cache[1].dtype == mx.float32
    assert mx.allclose(prefill, decoded, rtol=2e-2, atol=2e-2).item()


def test_puzzle_preserves_timestep_activation_precision():
    args = replace(
        puzzle_args(),
        num_hidden_layers=1,
        hybrid_override_pattern=["M"],
        layers_block_type=["mamba"],
        block_configs=[{"block_type": "mamba"}],
        time_step_min=0.001,
        time_step_limit=None,
    )
    mixer = nemotron_h.NemotronHMamba2Mixer(args)
    hidden = mx.zeros((1, 2, mixer.intermediate_size), dtype=mx.bfloat16)
    B = mx.zeros((1, 2, mixer.n_groups * mixer.ssm_state_size), dtype=mx.bfloat16)
    C = mx.zeros_like(B)
    dt = mx.zeros((1, 2, mixer.num_heads), dtype=mx.bfloat16)
    captured = {}

    def fake_ssm_update(*values, **kwargs):
        captured["hidden_dtype"] = values[0].dtype
        captured["B_dtype"] = values[2].dtype
        captured["C_dtype"] = values[3].dtype
        captured["dt_dtype"] = values[5].dtype
        captured["dt_bias_dtype"] = values[6].dtype
        captured["promote_dt"] = kwargs["promote_dt"]
        captured["time_step_limit"] = values[8]
        return values[0], None

    with patch.object(nemotron_h, "ssm_update", fake_ssm_update):
        output = mixer._ssm(hidden, B, C, dt, cache=None, mask=None)

    assert output.dtype == mx.bfloat16
    assert captured == {
        "hidden_dtype": mx.float32,
        "B_dtype": mx.float32,
        "C_dtype": mx.float32,
        "dt_dtype": mx.bfloat16,
        "dt_bias_dtype": mx.bfloat16,
        "promote_dt": False,
        "time_step_limit": (0.001, float("inf")),
    }


def test_official_source_model_prefix_is_remapped_to_backbone():
    model = nemotron_h.Model(puzzle_args())
    weights = {
        "model.embeddings.weight": mx.ones((128, 32)),
        "lm_head.weight": mx.ones((128, 32)),
    }

    sanitized = model.sanitize(weights)

    assert "model.embeddings.weight" not in sanitized
    assert "backbone.embeddings.weight" in sanitized


def test_official_source_experts_are_remapped_and_stacked():
    model = nemotron_h.Model(puzzle_args())
    weights = {}
    for expert in range(4):
        prefix = f"model.layers.0.mixer.experts.{expert}"
        weights[f"{prefix}.up_proj.weight"] = mx.full((32, 32), expert)
        weights[f"{prefix}.down_proj.weight"] = mx.full((32, 32), expert)

    sanitized = model.sanitize(weights)

    fc1 = sanitized["backbone.layers.0.mixer.switch_mlp.fc1.weight"]
    fc2 = sanitized["backbone.layers.0.mixer.switch_mlp.fc2.weight"]
    assert fc1.shape == (4, 32, 32)
    assert fc2.shape == (4, 32, 32)
    assert mx.array_equal(fc1[:, 0, 0], mx.arange(4)).item()
    assert not any("experts." in key for key in sanitized)


def test_strict_load_of_heterogeneous_quantized_checkpoint(tmp_path):
    args = puzzle_args()
    model = nemotron_h.Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())

    save_model(tmp_path, model)
    config = asdict(args)
    config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))

    loaded, _ = load_model(tmp_path, strict=True)
    logits = loaded(mx.array([[1, 2, 3]]))
    mx.eval(logits)

    assert loaded.layers[0].mixer.switch_mlp.fc1.output_dims == 32
    assert loaded.layers[1].mixer.switch_mlp.fc1.output_dims == 64
    assert logits.shape == (1, 3, 128)
