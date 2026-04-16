"""Residual Precision Streaming (RPS) for memory-efficient LLM inference.

Enables running larger models on smaller hardware by decomposing weights
into a low-bit base (always in memory) and a quantized residual that
boosts precision on demand.

Example:
    # Convert a model
    from mlx_lm.rps import convert, load_rps

    convert("mlx-community/Llama-3.1-70B-4bit",
            base_output="./llama-70b-rps-base",
            residual_output="./llama-70b-rps-residuals")

    # Load and generate
    model, tokenizer = load_rps(
        "./llama-70b-rps-base",
        "./llama-70b-rps-residuals",
        tier=2,
    )
"""

from .config import RPSConfig
from .convert import convert, compute_residuals
from .linear import RPSLinear
from .loader import load_rps
