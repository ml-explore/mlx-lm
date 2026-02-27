# Copyright © 2023-2024 Apple Inc.

import argparse
import re
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .utils import (
    QUANT_MODE_DEFAULTS,
    dequantize_model,
    load,
    quantize_model,
    save,
    upload_to_hub,
)


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module, group_size: int = 64
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:
    mode = "affine"
    high_bits = 6

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_4":
        low_bits = 3
        high_bits = 4
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError(f"Invalid quant recipe {recipe}")

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """
        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if (
            "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path
        ) and use_more_bits:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}

        return {"group_size": group_size, "bits": low_bits, "mode": mode}

    return mixed_quant_predicate


QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]

FLOAT_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}

QUANT_MODES = {
    mode: {"group_size": group_size, "bits": bits, "mode": mode}
    for mode, (group_size, bits) in QUANT_MODE_DEFAULTS.items()
    if mode != "affine"
}

ParsedOverride = tuple[re.Pattern, Union[int, str]]


def warn_mode_override_conflicts(
    q_mode: str,
    q_group_size: Optional[int],
    q_bits: Optional[int],
    default_group_size: int,
    default_bits: int,
) -> None:
    if q_mode == "affine":
        return

    conflicts = []
    if q_group_size is not None and q_group_size != default_group_size:
        conflicts.append(f"q-group-size={q_group_size}")
    if q_bits is not None and q_bits != default_bits:
        conflicts.append(f"q-bits={q_bits}")

    if conflicts:
        details = ", ".join(conflicts)
        print(
            f"[WARN] --q-mode {q_mode} default is "
            f"q-group-size={default_group_size}, q-bits={default_bits}; "
            f"received {details}. This may produce unexpected results."
        )


def warn_mixed_mode_overrides(
    q_mode: str, overrides: list[ParsedOverride]
) -> None:
    if q_mode == "affine":
        return
    if not any(isinstance(value, int) for _, value in overrides):
        return
    print(
        f"[WARN] Integer --q-override values force affine quantization on "
        f"matching layers. With --q-mode {q_mode}, this produces mixed "
        f"quantization modes."
    )


def parse_overrides(overrides: list[str]) -> list[ParsedOverride]:
    parsed = []
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(
                f"Invalid override '{entry}'. Expected PATTERN=VALUE"
            )
        pattern, value = entry.split("=", 1)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex in override '{pattern}': {e}")
        if value in FLOAT_DTYPES or value in QUANT_MODES:
            parsed.append((compiled, value))
        else:
            try:
                parsed.append((compiled, int(value)))
            except ValueError:
                valid = list(FLOAT_DTYPES) + list(QUANT_MODES)
                raise ValueError(
                    f"Invalid override value '{value}'. "
                    f"Expected an integer (bit width) or one of {valid}"
                )
    return parsed


def build_override_predicate(
    overrides: list[ParsedOverride],
    base_predicate: Optional[Callable[[str, nn.Module], Union[bool, dict]]],
    group_size: int,
    int_group_size: Optional[int] = None,
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    def predicate(path, module):
        for regex, value in overrides:
            if regex.search(path):
                if value in QUANT_MODES:
                    return dict(QUANT_MODES[value])
                if isinstance(value, str):
                    return False
                resolved_group_size = (
                    group_size if int_group_size is None else int_group_size
                )
                return {
                    "group_size": resolved_group_size,
                    "bits": value,
                    "mode": "affine",
                }
        if base_predicate is not None:
            return base_predicate(path, module)
        return True

    return predicate


def apply_float_overrides(model: nn.Module, overrides: list[ParsedOverride]) -> None:
    float_overrides = [(r, FLOAT_DTYPES[v]) for r, v in overrides if v in FLOAT_DTYPES]
    if not float_overrides:
        return

    def maybe_cast(path, value):
        # Only cast weights; biases are small and typically not quantized
        if not path.endswith(".weight"):
            return value
        if not mx.issubdtype(value.dtype, mx.floating):
            return value
        for regex, target_dtype in float_overrides:
            if regex.search(path):
                return value.astype(target_dtype)
        return value

    model.update(tree_map_with_path(maybe_cast, model.parameters()))


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: Optional[int] = None,
    q_bits: Optional[int] = None,
    q_mode: str = "affine",
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[
        Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]
    ] = None,
    trust_remote_code: bool = False,
    q_overrides: Optional[list[str]] = None,
):
    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    default_gs, default_bits = QUANT_MODE_DEFAULTS[q_mode]
    warn_mode_override_conflicts(q_mode, q_group_size, q_bits, default_gs, default_bits)
    q_group_size = default_gs if q_group_size is None else q_group_size
    q_bits = default_bits if q_bits is None else q_bits

    print("[INFO] Loading")
    model, tokenizer, config = load(
        hf_path,
        revision=revision,
        return_config=True,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        lazy=True,
    )

    if isinstance(quant_predicate, str):
        if q_mode != "affine":
            raise ValueError(f"Quant predicates only support 'affine' quantization.")
        quant_predicate = mixed_quant_predicate_builder(
            quant_predicate,
            model,
            q_group_size,
        )

    parsed_overrides = None
    if q_overrides:
        parsed_overrides = parse_overrides(q_overrides)
        warn_mixed_mode_overrides(q_mode, parsed_overrides)
        affine_group_size, _ = QUANT_MODE_DEFAULTS["affine"]
        int_override_group_size = (
            q_group_size if q_mode == "affine" else affine_group_size
        )
        quant_predicate = build_override_predicate(
            parsed_overrides,
            quant_predicate,
            q_group_size,
            int_group_size=int_override_group_size,
        )

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype is None and (text_config := config.get("text_config", None)):
        dtype = text_config.get("dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        model.update(tree_map_with_path(set_dtype, model.parameters()))

    # Apply float overrides last so per-layer dtype rules win over global --dtype.
    if parsed_overrides:
        apply_float_overrides(model, parsed_overrides)

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model, config = quantize_model(
            model,
            config,
            q_group_size,
            q_bits,
            mode=q_mode,
            quant_predicate=quant_predicate,
        )

    if dequantize:
        print("[INFO] Dequantizing")
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        model = dequantize_model(model)

    save(
        mlx_path,
        hf_path,
        model,
        tokenizer,
        config,
    )

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        default="affine",
        choices=list(QUANT_MODE_DEFAULTS),
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--q-override",
        help=(
            "Per-layer quantization override as PATTERN=VALUE (repeatable). "
            "PATTERN is a regex matched against the module path. "
            "VALUE is a bit width (int), dtype (float16, bfloat16, float32), "
            "or quant mode (mxfp4, nvfp4, mxfp8). Integer bit overrides are "
            "affine for matching layers."
        ),
        action="append",
        default=None,
        dest="q_overrides",
        metavar="PATTERN=VALUE",
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters. Defaults to config.json's `torch_dtype` or the current model weights dtype.",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Trust remote code when loading tokenizer.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.convert ...` directly is deprecated."
        " Use `mlx_lm.convert ...` or `python -m mlx_lm convert ...` instead."
    )
    main()
