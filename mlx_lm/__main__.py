# Copyright © 2025 Apple Inc.

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "quant.awq",
        "quant.dwq",
        "quant.dynamic_quant",
        "quant.gptq",
        "benchmark",
        "cache_prompt",
        "chat",
        "convert",
        "evaluate",
        "fuse",
        "generate",
        "lora",
        "perplexity",
        "server",
        "manage",
        "upload",
    }
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand in subcommands:
        submodule = importlib.import_module(f"mlx_lm.{subcommand}")
        submodule.main()
    elif subcommand == "--version":
        from mlx_lm import __version__

        print(__version__)
    else:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
