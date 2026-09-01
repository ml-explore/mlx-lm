import argparse
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


def local_experts(indices, num_experts, group: mx.distributed.Group):

    N = group.size()
    if num_experts % N != 0:
        raise ValueError(f"Cannot shard {num_experts} experts across {N} devices.")

    mask = (indices % N) == group.rank()

    return mx.where(mask, indices // N, 0), mask


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.sharding_group = None
        self.sharding = None

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
            scores = scores / scores.sum(axis=-1, keepdims=True) # normalise in the same way on each gpu
        if self.sharding == 'ep':
            inds, mask = local_experts(inds, self.num_experts, self.sharding_group)
            scores = mx.where(mask, scores, 0)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y

class Model(nn.Module):
    """One Qwen3-Next sparse MoE block: router and routed experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.mlp = Qwen3NextSparseMoeBlock(args)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


    def shard_tp(self, group: mx.distributed.Group):
        """Shards the model for tensor parallelism."""
        # The block sums the partial results at the end of the forward pass.
        self.mlp.sharding_group = group
        self.mlp.sharding = "tp"
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
        self.mlp.sharding_group = group
        self.mlp.sharding = "ep"
        shard_inplace(
            self.mlp.switch_mlp.gate_proj, "expert-parallel", group=group
        ) # [E, H, D] -> [E // N, H, D]
        shard_inplace(
            self.mlp.switch_mlp.down_proj, "expert-parallel", group=group
        ) # [E, D, H] -> [E // N, D, H]
        shard_inplace(
            self.mlp.switch_mlp.up_proj, "expert-parallel", group=group
        ) # [E, H, D] -> [E // N, H, D]


def barrier(group: mx.distributed.Group):
    """Waits for every rank, so they all start the same work together."""
    if group.size() > 1:
        mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))


def all_max(value: float, group: mx.distributed.Group) -> float:
    """The value of the slowest rank, which sets the step time."""
    if group.size() == 1:
        return value
    return mx.distributed.all_max(mx.array(value), group=group, stream=mx.cpu).item()


def bench_step(
    model,
    args,
    group: mx.distributed.Group,
    seq_len: int = 1,
    steps: int = 100,
    warmup: int = 5,
):
    xs = [
        mx.random.normal((1, seq_len, args.hidden_size)).astype(mx.bfloat16)
        for _ in range(steps)
    ]
    mx.eval(xs, model.parameters())

    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        for x in xs[:warmup]:
            mx.eval(model(x))

        barrier(group)
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
        dt = (time.perf_counter() - tic) / steps

    return all_max(dt, group)


def run_tests(group: mx.distributed.Group):
    """Checks the sharded block against the unsharded one on every rank.

    Every rank draws the same weights and the same input, so each can build
    the unsharded reference locally and compare it to what the sharded block
    returns after the real all_sum.
    """
    N = group.size()
    args = ModelArgs(
        hidden_size=64,
        moe_intermediate_size=128,
        num_experts=256,
        num_experts_per_tok=8,
    )

    inds = mx.array([[5, 12, 77, 130, 201, 3, 64, 99]])
    lidx, mask = local_experts(inds, args.num_experts, group)
    assert (lidx < args.num_experts // N).all().item()
    assert (mx.where(mask, group.rank() + lidx * N, inds) == inds).all().item()
    claims = mx.distributed.all_sum(mask.astype(mx.int32), group=group)
    assert (claims == 1).all().item(), claims.tolist()

    mx.random.seed(0)
    x = mx.random.normal((1, 5, args.hidden_size))
    reference = Model(args)
    ref = reference(x)

    for mode in ("tp", "ep"):
        experts = args.num_experts // N if mode == "ep" else args.num_experts
        model = Model(args)
        model.update(reference.parameters())
        getattr(model, f"shard_{mode}")(group)
        assert model.mlp.switch_mlp.gate_proj.weight.shape[0] == experts

        err = mx.abs(model(x) - ref).max().item()
        assert err < 1e-5, (mode, err)
        if group.rank() == 0:
            print(f"{mode}: matches unsharded on {N} rank(s), max err {err:.2e}")

    if group.rank() == 0:
        print("local_experts: every expert owned by exactly one rank")


def main():
    parser = argparse.ArgumentParser(description="Benchmark one sparse MoE block")
    parser.add_argument("--sharding", choices=["none", "tp", "ep"], default="none")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--test", action="store_true")
    cli = parser.parse_args()

    # The same seed gives every rank the same weights and the same inputs.
    mx.random.seed(0)
    group = mx.distributed.init()

    if cli.test:
        run_tests(group)
        return

    args = ModelArgs()
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    if cli.sharding == "tp":
        model.shard_tp(group)
    elif cli.sharding == "ep":
        model.shard_ep(group)
    mx.eval(model.parameters())
    barrier(group)

    dt = bench_step(
        model,
        args,
        group,
        seq_len=cli.seq_len,
        steps=cli.steps,
        warmup=cli.warmup,
    )
    # memory = all_max(mx.get_peak_memory() / 1e9, group)
    if group.rank() == 0:
        print(
            f"{cli.sharding} on {group.size()} rank(s), seq_len {cli.seq_len}: "
            f"{dt * 1e3:.3f} ms/step"
        )


if __name__ == "__main__":
    main()
