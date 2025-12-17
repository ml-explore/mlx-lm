# Copyright © 2024 Apple Inc.

import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_lm"
sys.path.append(str(package_dir))

from _version import __version__

MIN_MLX_VERSION = "0.29.2"

setup(
    name="mlx-lm",
    version=__version__,
    description="LLMs with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-lm",
    license="MIT",
    install_requires=[
        f"mlx>={MIN_MLX_VERSION}; platform_system == 'Darwin'",
        "numpy",
        "transformers==5.0.0rc1",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "jinja2",
    ],
    packages=["mlx_lm", "mlx_lm.models", "mlx_lm.quant", "mlx_lm.tuner"],
    python_requires=">=3.8",
    extras_require={
        "test": ["datasets", "lm-eval"],
        "train": ["datasets", "tqdm"],
        "evaluate": ["lm-eval", "tqdm"],
        "cuda": [f"mlx[cuda]>={MIN_MLX_VERSION}"],
        "cpu": [f"mlx[cpu]>={MIN_MLX_VERSION}"],
    },
    entry_points={
        "console_scripts": [
            "mlx_lm.awq = mlx_lm.quant.awq:main",
            "mlx_lm.dwq = mlx_lm.quant.dwq:main",
            "mlx_lm.dynamic_quant = mlx_lm.quant.dynamic_quant:main",
            "mlx_lm.gptq = mlx_lm.quant.gptq:main",
            "mlx_lm.benchmark = mlx_lm.benchmark:main",
            "mlx_lm.cache_prompt = mlx_lm.cache_prompt:main",
            "mlx_lm.chat = mlx_lm.chat:main",
            "mlx_lm.convert = mlx_lm.convert:main",
            "mlx_lm.evaluate = mlx_lm.evaluate:main",
            "mlx_lm.fuse = mlx_lm.fuse:main",
            "mlx_lm.generate = mlx_lm.generate:main",
            "mlx_lm.lora = mlx_lm.lora:main",
            "mlx_lm.perplexity = mlx_lm.perplexity:main",
            "mlx_lm.server = mlx_lm.server:main",
            "mlx_lm.manage = mlx_lm.manage:main",
            "mlx_lm.upload = mlx_lm.upload:main",
        ]
    },
)
