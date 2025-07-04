import argparse

import mlx.core as mx

from mlx_lm.convert import convert


def mixed_quant(layer_path, layer, cfg):
    if "embedding" in layer_path:
        return {"group_size": 32, "bits": 8}
    return hasattr(layer, "to_quantized")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Quantize the AFM according to its original quantization"
    )
    parser.add_argument("source", help="The mlx model containing the fp32 weights")
    parser.add_argument("destination", help="The folder to save the quantized model to")
    parser.add_argument("--copy-adapters", action="store_true")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="float32"
    )
    args = parser.parse_args(argv)

    convert(
        args.source,
        args.destination,
        quantize=True,
        q_group_size=128,
        q_bits=2,
        dtype=getattr(mx, args.dtype),
        quant_predicate=mixed_quant,
    )


if __name__ == "__main__":
    main(None)
