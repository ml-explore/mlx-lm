# Copyright © 2026 Apple Inc.

import argparse
from pathlib import Path

from mlx.utils import tree_flatten

from .tuner.utils import quantize_lora_layers, save_quantized_adapter
from .utils import load


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize LoRA adapter matrices for direct MLX inference."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model path or Hugging Face repository.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to an MLX LoRA adapter.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory for the quantized adapter.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=(2, 3, 4, 5, 6, 8),
        default=8,
        help="Bits per LoRA weight. Default: 8.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size for LoRA A. Default: 64.",
    )
    parser.add_argument(
        "--rank-group-size",
        type=int,
        default=32,
        help="Quantization group size for LoRA B. Default: 32.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the model repository.",
    )
    parser.add_argument(
        "--allow-larger",
        action="store_true",
        help="Save even if quantization increases adapter parameter memory.",
    )
    return parser


def adapter_parameter_bytes(model) -> int:
    return sum(
        value.nbytes
        for name, value in tree_flatten(model.parameters())
        if any(part.startswith("lora_") for part in name.split("."))
    )


def main() -> None:
    args = configure_parser().parse_args()
    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    model, _ = load(
        args.model,
        adapter_path=str(adapter_path),
        trust_remote_code=args.trust_remote_code,
    )
    float_parameter_bytes = adapter_parameter_bytes(model)
    quantize_lora_layers(
        model,
        group_size=args.group_size,
        bits=args.bits,
        rank_group_size=args.rank_group_size,
    )
    quantized_parameter_bytes = adapter_parameter_bytes(model)
    if quantized_parameter_bytes >= float_parameter_bytes and not args.allow_larger:
        raise ValueError(
            "Quantization does not reduce adapter parameter memory "
            f"({float_parameter_bytes} -> {quantized_parameter_bytes} bytes). "
            "Use a lower bit width, a smaller rank group size, or "
            "--allow-larger to override."
        )
    save_quantized_adapter(model, str(adapter_path), str(output_path))

    source_size = (adapter_path / "adapters.safetensors").stat().st_size
    output_size = (output_path / "adapters.safetensors").stat().st_size
    reduction = 1 - output_size / source_size
    parameter_reduction = 1 - quantized_parameter_bytes / float_parameter_bytes
    print(f"Saved quantized adapter to {output_path}")
    print(
        "Adapter parameter memory: "
        f"{float_parameter_bytes / 1e6:.2f} MB -> "
        f"{quantized_parameter_bytes / 1e6:.2f} MB "
        f"({parameter_reduction:.2%} reduction)"
    )
    print(f"Adapter size: {source_size / 1e6:.2f} MB -> {output_size / 1e6:.2f} MB")
    print(f"Size reduction: {reduction:.2%}")


if __name__ == "__main__":
    main()
