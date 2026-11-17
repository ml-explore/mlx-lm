# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import nemotron_h, ssm
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


# Mirrors the layout of the Nemotron 3.5 Bolt ``llm_config``: the expert
# count, expert width and top-k are set per MoE block only (they are
# "blockwise_members" and absent from the global config), the latent size
# is global, and the time step limit uses transformers' encoding of inf.
BOLT_LIKE_BLOCKS = [
    {"block_type": "mamba"},
    {
        "block_type": "moe",
        "moe_intermediate_size": 32,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
    },
    {"block_type": "attention"},
    {
        "block_type": "moe",
        "moe_intermediate_size": 96,
        "n_routed_experts": 6,
        "num_experts_per_tok": 2,
    },
    {"block_type": "mamba"},
    {
        "block_type": "moe",
        "moe_intermediate_size": 64,
        "n_routed_experts": 8,
        "num_experts_per_tok": 3,
    },
]


def bolt_like_config():
    return {
        "model_type": "nemotron_h_puzzle",
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "blockwise_members": [
            "block_type",
            "n_routed_experts",
            "moe_intermediate_size",
            "num_experts_per_tok",
        ],
        "block_configs": [dict(b) for b in BOLT_LIKE_BLOCKS],
        "layers_block_type": [b["block_type"] for b in BOLT_LIKE_BLOCKS],
        "mtp_block_configs": [],
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 64,
        "max_position_embeddings": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "attention_bias": False,
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "mamba_proj_bias": False,
        "ssm_state_size": 8,
        "conv_kernel": 4,
        "n_groups": 2,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "moe_latent_size": 32,
        "moe_shared_expert_intermediate_size": 64,
        "n_shared_experts": 1,
        "num_experts": 512,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 5.0,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "time_step_min": 0.001,
        "time_step_max": 0.1,
        "tie_word_embeddings": False,
    }


def to_hf_checkpoint(model, prefix="language_model."):
    """Rename MLX weights to the original multimodal checkpoint layout."""
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:
            layer_prefix, rest = k.split(".switch_mlp.")
            proj = {"fc1.weight": "up_proj", "fc2.weight": "down_proj"}[rest]
            for e in range(v.shape[0]):
                weights[f"{prefix}{layer_prefix}.experts.{e}.{proj}.weight"] = v[e]
        elif "conv1d.weight" in k:
            weights[prefix + k] = v.moveaxis(2, 1)
        else:
            weights[prefix + k] = v
    # Components of the omni checkpoint that are not part of the text model.
    weights["vision_model.radio_model.model.patch_generator.embedder.weight"] = (
        mx.zeros((8, 4))
    )
    weights["vision_model.summary_idxs"] = mx.zeros((3,), dtype=mx.int64)
    weights["mlp1.0.weight"] = mx.zeros((8,))
    weights["mlp1.1.weight"] = mx.zeros((8, 8))
    return weights


class TestNemotronHPuzzle(unittest.TestCase):
    def test_puzzle_model_type_remaps_to_nemotron_h(self):
        model_cls, args_cls = _get_classes({"model_type": "nemotron_h_puzzle"})
        self.assertIs(model_cls, nemotron_h.Model)
        self.assertIs(args_cls, nemotron_h.ModelArgs)

    def test_puzzle_quantization_preserves_output_head(self):
        args = puzzle_args()
        model = nemotron_h.Model(args)

        quantize_model(model, asdict(args), group_size=32, bits=4)

        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertFalse(hasattr(model.lm_head, "scales"))
        self.assertTrue(hasattr(model.layers[0].mixer.switch_mlp.fc1, "scales"))

    def test_official_config_derives_layer_count(self):
        config = asdict(puzzle_args())
        config.pop("num_hidden_layers")

        args = nemotron_h.ModelArgs.from_dict(config)

        self.assertEqual(args.num_hidden_layers, len(config["block_configs"]))

    def test_puzzle_layers_use_heterogeneous_moe_dimensions_and_top_k(self):
        model = nemotron_h.Model(puzzle_args())

        first, second = model.layers
        self.assertEqual(first.mixer.switch_mlp.fc1.output_dims, 32)
        self.assertEqual(second.mixer.switch_mlp.fc1.output_dims, 64)
        self.assertEqual(first.mixer.gate.top_k, 1)
        self.assertEqual(second.mixer.gate.top_k, 2)

        logits = model(mx.array([[1, 2, 3]]))
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, 128))

    def test_mamba_prefill_matches_cached_decode(self):
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

        self.assertEqual(cache[1].dtype, mx.float32)
        self.assertTrue(mx.allclose(prefill, decoded, rtol=2e-2, atol=2e-2).item())

    def test_puzzle_preserves_timestep_activation_precision(self):
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

        self.assertEqual(output.dtype, mx.bfloat16)
        self.assertEqual(
            captured,
            {
                "hidden_dtype": mx.float32,
                "B_dtype": mx.float32,
                "C_dtype": mx.float32,
                "dt_dtype": mx.bfloat16,
                "dt_bias_dtype": mx.bfloat16,
                "promote_dt": False,
                "time_step_limit": (0.001, float("inf")),
            },
        )

    def test_native_timestep_keeps_state_transition_in_float32(self):
        hidden = mx.array([[[[0.5]], [[-0.25]]]], dtype=mx.float32)
        A_log = mx.array([1.234567], dtype=mx.float32)
        B = mx.array([[[[0.75]], [[-0.5]]]], dtype=mx.float32)
        C = mx.array([[[[0.25]], [[1.5]]]], dtype=mx.float32)
        D = mx.array([0.125], dtype=mx.float32)
        dt = mx.array([[[0.125], [0.25]]], dtype=mx.bfloat16)
        dt_bias = mx.array([-0.1], dtype=mx.bfloat16)
        initial_state = mx.ones((1, 1, 1, 1), dtype=mx.float32)

        output, final_state = ssm.ssm_attn(
            hidden,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            initial_state,
            (0.001, float("inf")),
            promote_dt=False,
        )

        reference_dt = nn.softplus(dt + dt_bias).astype(mx.float32)
        reference_A = -mx.exp(A_log)
        reference_state = initial_state
        reference_output = []
        for position in range(hidden.shape[1]):
            delta = reference_dt[:, position]
            reference_state = reference_state * mx.exp(
                delta[:, :, None, None] * reference_A[None, :, None, None]
            )
            reference_state = reference_state + (
                delta[:, :, None, None]
                * hidden[:, position, :, :, None]
                * B[:, position, :, None, :]
            )
            current = (reference_state * C[:, position, :, None, :]).sum(axis=-1)
            current = current + D[None, :, None] * hidden[:, position]
            reference_output.append(current)
        reference_output = mx.stack(reference_output, axis=1)
        mx.eval(output, final_state, reference_output, reference_state)

        self.assertTrue(
            mx.allclose(output, reference_output, rtol=1e-5, atol=1e-5).item()
        )
        self.assertTrue(
            mx.allclose(final_state, reference_state, rtol=1e-5, atol=1e-5).item()
        )

    def test_explicit_time_step_limit_is_preserved(self):
        explicit_limit = (0.001, 100.0)
        for model_type in ("nemotron_h", "nemotron_h_puzzle"):
            args = replace(
                puzzle_args(),
                model_type=model_type,
                time_step_min=0.002,
                time_step_limit=explicit_limit,
            )
            self.assertEqual(args.time_step_limit, explicit_limit)

    def test_puzzle_precision_changes_do_not_change_base_nemotron_h(self):
        common = dict(
            num_hidden_layers=1,
            hybrid_override_pattern=["M"],
            layers_block_type=["mamba"],
            use_bias=True,
            mamba_proj_bias=False,
            time_step_min=0.001,
            time_step_limit=None,
        )
        puzzle = replace(
            puzzle_args(), block_configs=[{"block_type": "mamba"}], **common
        )
        base = replace(
            puzzle_args(), model_type="nemotron_h", block_configs=None, **common
        )
        self.assertTrue(puzzle.is_puzzle)
        self.assertFalse(base.is_puzzle)
        puzzle_mixer = nemotron_h.NemotronHMamba2Mixer(puzzle)
        base_mixer = nemotron_h.NemotronHMamba2Mixer(base)
        puzzle_block = nemotron_h.NemotronHBlock(puzzle, "M")
        base_block = nemotron_h.NemotronHBlock(base, "M")
        router_dtypes = []

        def capture_router_dtype(gates, *_args):
            router_dtypes.append(gates.dtype)
            return None, None

        puzzle_gate = nemotron_h.MoEGate(puzzle)
        base_gate = nemotron_h.MoEGate(base)
        puzzle_gate.weight = puzzle_gate.weight.astype(mx.bfloat16)
        base_gate.weight = base_gate.weight.astype(mx.bfloat16)
        with patch.object(nemotron_h, "group_expert_select", capture_router_dtype):
            puzzle_gate(mx.ones((1, puzzle.hidden_size), dtype=mx.bfloat16))
            base_gate(mx.ones((1, base.hidden_size), dtype=mx.bfloat16))

        self.assertIn("bias", puzzle_mixer.in_proj.parameters())
        self.assertIn("bias", puzzle_mixer.out_proj.parameters())
        self.assertNotIn("bias", base_mixer.in_proj.parameters())
        self.assertNotIn("bias", base_mixer.out_proj.parameters())
        self.assertIsInstance(puzzle_block.norm, nemotron_h.NemotronHRMSNorm)
        self.assertIsInstance(base_block.norm, nn.RMSNorm)
        self.assertEqual(router_dtypes, [mx.float32, mx.bfloat16])
        self.assertEqual(puzzle.time_step_limit, (0.001, float("inf")))
        self.assertEqual(base.time_step_limit, (0.0, float("inf")))

    def test_block_configs_enable_puzzle_paths_without_puzzle_model_type(self):
        # Some converted Puzzle checkpoints keep model_type "nemotron_h".
        config = bolt_like_config()
        config["model_type"] = "nemotron_h"
        model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(config))
        self.assertIsInstance(model.backbone.norm_f, nemotron_h.NemotronHRMSNorm)
        self.assertTrue(model.layers[0].mixer.is_puzzle)
        self.assertFalse(model.quant_predicate("lm_head", model.lm_head))

    def test_official_source_model_prefix_is_remapped_to_backbone(self):
        model = nemotron_h.Model(puzzle_args())
        weights = {
            "model.embeddings.weight": mx.ones((128, 32)),
            "lm_head.weight": mx.ones((128, 32)),
        }

        sanitized = model.sanitize(weights)

        self.assertNotIn("model.embeddings.weight", sanitized)
        self.assertIn("backbone.embeddings.weight", sanitized)

    def test_official_source_experts_are_remapped_and_stacked(self):
        model = nemotron_h.Model(puzzle_args())
        weights = {}
        for expert in range(4):
            prefix = f"model.layers.0.mixer.experts.{expert}"
            weights[f"{prefix}.up_proj.weight"] = mx.full((32, 32), expert)
            weights[f"{prefix}.down_proj.weight"] = mx.full((32, 32), expert)

        sanitized = model.sanitize(weights)

        fc1 = sanitized["backbone.layers.0.mixer.switch_mlp.fc1.weight"]
        fc2 = sanitized["backbone.layers.0.mixer.switch_mlp.fc2.weight"]
        self.assertEqual(fc1.shape, (4, 32, 32))
        self.assertEqual(fc2.shape, (4, 32, 32))
        self.assertTrue(mx.array_equal(fc1[:, 0, 0], mx.arange(4)).item())
        self.assertFalse(any("experts." in key for key in sanitized))

    def test_strict_load_of_heterogeneous_quantized_checkpoint(self):
        args = puzzle_args()
        model = nemotron_h.Model(args)
        nn.quantize(model, group_size=32, bits=4)
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_model(tmp_path, model)
            config = asdict(args)
            config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
            (tmp_path / "config.json").write_text(json.dumps(config))

            loaded, _ = load_model(tmp_path, strict=True)
        logits = loaded(mx.array([[1, 2, 3]]))
        mx.eval(logits)

        self.assertEqual(loaded.layers[0].mixer.switch_mlp.fc1.output_dims, 32)
        self.assertEqual(loaded.layers[1].mixer.switch_mlp.fc1.output_dims, 64)
        self.assertEqual(logits.shape, (1, 3, 128))

    # Nemotron 3.5 Bolt: per-block expert counts and multimodal key layout.

    def test_bolt_config_per_block_expert_counts(self):
        config = bolt_like_config()
        args = nemotron_h.ModelArgs.from_dict(config)
        self.assertIsNone(args.n_routed_experts)
        self.assertEqual(args.hybrid_override_pattern, ["M", "E", "*", "E", "M", "E"])
        self.assertEqual(args.time_step_limit, (0.0, float("inf")))

        model = nemotron_h.Model(args)
        for layer, block in zip(model.layers, BOLT_LIKE_BLOCKS):
            if block["block_type"] != "moe":
                self.assertFalse(hasattr(layer.mixer, "switch_mlp"))
                continue
            mixer = layer.mixer
            n, inter = block["n_routed_experts"], block["moe_intermediate_size"]
            self.assertEqual(mixer.gate.n_routed_experts, n)
            self.assertEqual(mixer.gate.top_k, block["num_experts_per_tok"])
            self.assertEqual(mixer.gate.weight.shape, (n, 64))
            self.assertEqual(mixer.gate.e_score_correction_bias.shape, (n,))
            self.assertEqual(mixer.switch_mlp.fc1.weight.shape, (n, inter, 32))
            self.assertEqual(mixer.switch_mlp.fc2.weight.shape, (n, 32, inter))
            self.assertEqual(mixer.fc1_latent_proj.weight.shape, (32, 64))
            self.assertEqual(mixer.fc2_latent_proj.weight.shape, (64, 32))
            self.assertEqual(mixer.shared_experts.up_proj.weight.shape, (64, 64))
        # One cache per mamba / attention layer.
        self.assertEqual(len(model.make_cache()), 3)

    def test_bolt_missing_expert_count_is_rejected(self):
        config = bolt_like_config()
        del config["block_configs"][3]["n_routed_experts"]
        with self.assertRaisesRegex(ValueError, "n_routed_experts"):
            nemotron_h.ModelArgs.from_dict(config)

    def test_bolt_global_expert_count_is_a_fallback(self):
        config = bolt_like_config()
        config["n_routed_experts"] = 4
        for block in config["block_configs"]:
            block.pop("n_routed_experts", None)
        model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(config))
        self.assertEqual(
            [
                l.mixer.gate.n_routed_experts
                for l in model.layers
                if l.block_type == "E"
            ],
            [4, 4, 4],
        )

    def test_bolt_forward_and_cached_decode(self):
        mx.random.seed(0)
        model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(bolt_like_config()))
        model.set_dtype(mx.float32)
        tokens = mx.array([[3, 17, 42, 5, 99, 7]])

        # On the CPU so the single-token SSM step matches the full-sequence
        # path to float32 precision (the GPU step kernel is not bit-equal).
        with mx.stream(mx.cpu):
            full = model(tokens)
            cache = model.make_cache()
            prefill = model(tokens[:, :4], cache=cache)
            step1 = model(tokens[:, 4:5], cache=cache)
            step2 = model(tokens[:, 5:6], cache=cache)
            mx.eval(full, prefill, step1, step2)

        self.assertEqual(full.shape, (1, 6, 128))
        self.assertEqual(step2.shape, (1, 1, 128))
        incremental = mx.concatenate([prefill, step1, step2], axis=1)
        self.assertTrue(mx.allclose(full, incremental, atol=1e-3, rtol=1e-3).item())

    def test_bolt_sanitize_multimodal_checkpoint_round_trip(self):
        mx.random.seed(1)
        args = nemotron_h.ModelArgs.from_dict(bolt_like_config())
        source = nemotron_h.Model(args)
        expected = dict(tree_flatten(source.parameters()))
        checkpoint = to_hf_checkpoint(source)
        # The original checkpoint stores per-expert tensors.
        self.assertIn(
            "language_model.backbone.layers.3.mixer.experts.5.up_proj.weight",
            checkpoint,
        )

        target = nemotron_h.Model(args)
        sanitized = target.sanitize(checkpoint)

        self.assertEqual(set(sanitized), set(expected))
        for k, v in expected.items():
            self.assertEqual(sanitized[k].shape, v.shape, k)
            self.assertTrue(mx.array_equal(sanitized[k], v).item(), k)
        target.load_weights(list(sanitized.items()), strict=True)

        tokens = mx.array([[1, 2, 3, 4]])
        self.assertTrue(mx.allclose(source(tokens), target(tokens)).item())

    def test_bolt_sanitize_is_idempotent_for_converted_weights(self):
        args = nemotron_h.ModelArgs.from_dict(bolt_like_config())
        model = nemotron_h.Model(args)
        weights = dict(tree_flatten(model.parameters()))
        sanitized = model.sanitize(dict(weights))
        self.assertEqual(set(sanitized), set(weights))
        for k, v in weights.items():
            self.assertEqual(sanitized[k].shape, v.shape, k)

    def test_bolt_quantize_and_reload(self):
        config = bolt_like_config()
        model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(config))
        _, config = quantize_model(model, config, group_size=32, bits=4)
        self.assertFalse(hasattr(model.lm_head, "scales"))
        self.assertTrue(hasattr(model.layers[5].mixer.switch_mlp.fc1, "scales"))
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmp:
            save_model(Path(tmp), model)
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            loaded, _ = load_model(Path(tmp), strict=True)
        tokens = mx.array([[1, 2, 3]])
        self.assertTrue(mx.allclose(model(tokens), loaded(tokens)).item())


if __name__ == "__main__":
    unittest.main()
