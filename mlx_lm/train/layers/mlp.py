# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.train.args import ModelArgs


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        # TODO: add the mixture of experts block
        raise NotImplementedError("sparse_moe is not implemented yet")


MLP_TYPES = {
    "mlp": MLP,
    "sparse_moe": SparseMoeBlock,
}


def build_mlp(args: ModelArgs, mlp_type: str) -> nn.Module:
    if mlp_type not in MLP_TYPES:
        raise ValueError(
            f"unknown mlp {mlp_type!r}; expected one of {', '.join(MLP_TYPES)}"
        )
    return MLP_TYPES[mlp_type](args)
