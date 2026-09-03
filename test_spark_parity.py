"""Numerical parity test: HF reference (torch) vs MLX port of Spark-X2.5.

Builds a small model with the same architectural features (hybrid SWA/full
attention, partial rotary, headwise gate) and checks max_abs_diff < 1e-3.
"""

import json
import sys

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, "/tmp")
from spark_ref_test.configuration_spark import Spark2_5Config  # noqa: E402
from spark_ref_test.modeling_spark import Spark2_5ForCausalLM  # noqa: E402

sys.path.insert(0, "/Users/dog/spark-mlx-port/mlx-lm")
from mlx_lm.utils import _get_classes  # noqa: E402


def make_small_config():
    # Same architectural features as the real model, small sizes for speed.
    return dict(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        sliding_window=8,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={
            "full_attention": {"rope_theta": 5000000.0, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
    )


def seed_mlx_from_torch(mlx_model, torch_model):
    """Copy torch state dict values into the MLX model."""
    torch_sd = torch_model.state_dict()

    # Build a flat MLX target dict mapping HF key -> MLX key
    # MLX uses model.layers.X.self_attn.q_k_v_proj.weight (same as HF)
    # Only difference: model.embedding.weight -> model.embed_tokens.weight
    torch_to_mlx = {}
    for k in torch_sd:
        if k == "model.embedding.weight":
            torch_to_mlx[k] = "model.embed_tokens.weight"
        elif k.startswith("lm_head."):
            continue  # tied
        else:
            torch_to_mlx[k] = k

    # Get MLX flat params
    mlx_flat = dict(tree_flatten(mlx_model.parameters()))
    torch_flat = {}
    for k, v in torch_sd.items():
        mk = torch_to_mlx.get(k)
        if mk is None or mk not in mlx_flat:
            print(f"  skip: {k} -> {mk}")
            continue
        torch_flat[mk] = mx.array(np.asarray(v.detach().float().numpy()))

    # Update only the matching keys
    mlx_flat.update(torch_flat)
    mlx_model.update(tree_unflatten(list(mlx_flat.items())))


def run_mlx(mlx_model, tokens, use_cache=False):
    t = mx.array([tokens])
    if use_cache:
        cache = mlx_model.make_cache()
        # Prefill
        _ = mlx_model(t, cache=cache)
        # One more decode step
        logits = mlx_model(mx.array([[tokens[-1]]]), cache=cache)
        return np.asarray(logits[0, -1])
    logits = mlx_model(t)
    return np.asarray(logits[0])


def run_torch(torch_model, tokens, extend=None):
    input_ids = torch.tensor([tokens + (extend or [])])
    with torch.no_grad():
        out = torch_model(input_ids=input_ids)
    if extend:
        return out.logits[0, -1].float().numpy()
    return out.logits[0].float().numpy()


def main():
    cfg = make_small_config()
    hf_cfg = Spark2_5Config(**cfg)
    torch_model = Spark2_5ForCausalLM(hf_cfg)
    torch_model.eval()

    config = {"model_type": "spark2_5", **cfg}
    cls, args_cls = _get_classes(config)
    args = args_cls.from_dict(config)
    mlx_model = cls(args)

    seed_mlx_from_torch(mlx_model, torch_model)

    tokens = [3, 7, 11, 2, 9]
    print("=== prefill (no cache) ===")
    mlx_logits = run_mlx(mlx_model, tokens)
    torch_logits = run_torch(torch_model, tokens)
    diff = np.abs(mlx_logits - torch_logits)
    print(f"  max_abs_diff: {diff.max():.6f}")
    print(f"  mean_abs_diff: {diff.mean():.6f}")
    assert diff.max() < 1e-3, f"prefill diff too large: {diff.max()}"

    print("=== decode with cache ===")
    mlx_logits2 = run_mlx(mlx_model, tokens, use_cache=True)
    torch_logits2 = run_torch(torch_model, tokens, extend=[tokens[-1]])
    diff2 = np.abs(mlx_logits2 - torch_logits2)
    print(f"  max_abs_diff: {diff2.max():.6f}")
    print(f"  mean_abs_diff: {diff2.mean():.6f}")
    assert diff2.max() < 1e-3, f"decode diff too large: {diff2.max()}"

    print("PARITY TEST PASSED")


if __name__ == "__main__":
    main()