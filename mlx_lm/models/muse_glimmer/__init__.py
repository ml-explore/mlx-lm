# Copyright © 2026 Apple Inc.
# Muse Glimmer (Meta, 2026-08-10) — text-tower support for mlx-lm.
#
# Glimmer is multimodal; this implements the language model only, which is what
# text/agentic inference needs. Architecture verified against Meta's reference
# transformers implementation (transformers 5.15 modeling_muse_glimmer.py).
#
# Glimmer-specific details (vs a standard Gemma/Llama block):
#   * gated attention output: attn * sigmoid(gate_proj(x)) before o_proj
#   * scaleless RMSNorm on Q and K, then Q scaled by qk_scale_factor
#   * [local, local, local, global] layer pattern; RoPE on local layers only,
#     global layers run NoPE (no positional embedding)
#   * scaleless RMSNorm applied to the token embeddings
#   * "centered" RMSNorm (output = norm(x) * (1 + weight)) on the four
#     per-block norms; standard RMSNorm on the final norm

from mlx_lm.models.muse_glimmer.config import ModelArgs
from mlx_lm.models.muse_glimmer.model import Model

__all__ = [
    "Model",
    "ModelArgs",
]
