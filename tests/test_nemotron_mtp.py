# Copyright © 2026 Apple Inc.

import pytest

from mlx_lm.models.nemotron_h import ModelArgs


def _args(pattern):
    return ModelArgs(
        model_type="nemotron_h",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=3,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=False,
        hybrid_override_pattern=["*", "M"],
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern=pattern,
    )


def test_nemotron_mtp_accepts_supported_attention_then_expert_pattern():
    assert _args("*E")._mtp_pattern == ["*", "E"]


@pytest.mark.parametrize("pattern", ["E*", "*", "EE", "M*"])
def test_nemotron_mtp_rejects_unsupported_patterns(pattern):
    with pytest.raises(ValueError, match="supports only the '\\*E' block pattern"):
        _args(pattern)
