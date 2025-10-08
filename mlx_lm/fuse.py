import argparse
from pathlib import Path

from mlx.utils import tree_flatten, tree_unflatten

from .gguf import convert_to_gguf
from .tuner.utils import dequantize, load_adapters
from .utils import (
    load,
    save,
    upload_to_hub,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse fine-tuned adapters into the base model."
    )
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters",
        help="Path to the trained adapter weights and config.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--de-quantize",
        help="Generate a de-quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--export-gguf",
        help="Export model weights in GGUF format.",
        action="store_true",
    )
    parser.add_argument(
        "--gguf-path",
        help="Path to save the exported GGUF format model weights. Default is ggml-model-f16.gguf.",
        default="ggml-model-f16.gguf",
        type=str,
    )
    return parser.parse_args()


def main() -> None:
    print("Loading pretrained model")
    args = parse_arguments()

    model, tokenizer, config = load(
        args.model, adapter_path=args.adapter_path, return_config=True
    )

    fused_linears = [
        (n, m.fuse(de_quantize=args.de_quantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if args.de_quantize:
        print("De-quantizing model")
        model = dequantize(model)
        config.pop("quantization", None)

    save_path = Path(args.save_path)
    save(
        save_path,
        args.model,
        model,
        tokenizer,
        config,
        donate_model=False,
    )

    if args.export_gguf:
        model_type = config["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        weights = dict(tree_flatten(model.parameters()))
        convert_to_gguf(save_path, weights, config, str(save_path / args.gguf_path))

    if args.upload_repo is not None:
        upload_to_hub(args.save_path, args.upload_repo)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.fuse...` directly is deprecated."
        " Use `mlx_lm.fuse...` or `python -m mlx_lm fuse ...` instead."
    )
    main()
