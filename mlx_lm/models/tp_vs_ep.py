import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, sum_gradients

from mlx_lm.generate import generation_stream, wired_limit
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    # Defaults are the Qwen3-Next-80B-A3B MoE settings.
    model_type: str = "tp_vs_ep"
    hidden_size: int = 6144
    moe_intermediate_size: int = 2048
    shared_expert_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        if self.sharding_group is not None:
            if self.sharding == 'tp':
                x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x) # router always duplicated 
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y    

class Model(nn.Module):
    """One Qwen3-Next sparse MoE block: router, experts and shared expert."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.mlp = Qwen3NextSparseMoeBlock(args)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


    def shard_tp(self, group: mx.distributed.Group):
        """Shards the model for tensor parallelism."""
        shard_inplace(
            self.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
        )
        shard_inplace(
            self.mlp.shared_expert.down_proj, "sharded-to-all", group=group
        )
        shard_inplace(
            self.mlp.shared_expert.up_proj, "all-to-sharded", group=group
        )
        shard_inplace(
            self.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
        )
        shard_inplace(
            self.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
        )
        shard_inplace(
            self.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
        )

    def shard_ep(self, group: mx.distributed.Group):
        """Shards the model for expert parallelism."""
        pass


def bench_step(model, args, seq_len: int = 1, steps: int = 100, warmup: int = 5):
    """Times one MoE step the way generate_step runs it.

    Same setup as generation: batch 1, the generation stream, a wired limit for
    the weights, one step of look-ahead and one sync per step. A new input each
    step keeps the routing random, like real decoding.
    """
    xs = [
        mx.random.normal((1, seq_len, args.hidden_size)).astype(mx.bfloat16)
        for _ in range(steps)
    ]
    mx.eval(xs, model.parameters())

    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        for x in xs[:warmup]:
            mx.eval(model(x))

        mx.reset_peak_memory()
        y = model(xs[0])
        mx.async_eval(y)
        tic = time.perf_counter()
        for x in xs[1:]:
            next_y = model(x)  # build the next graph while this step runs
            mx.async_eval(next_y)
            mx.eval(y)  # the sync that `y.item()` does in generate_step
            y = next_y
        mx.eval(y)
        return (time.perf_counter() - tic) / steps


if __name__ == "__main__":
    args = ModelArgs()
    model = Model(args)
    model.set_dtype(mx.bfloat16)

    dt = bench_step(model, args)
    print(f"{dt * 1e3:.3f} ms/step, peak memory {mx.get_peak_memory() / 1e9:.3f} GB")
