# Copyright © 2025 Apple Inc.

import mlx.core as mx
from mlx.utils import tree_flatten


class PipelineMixin:
    def __init__(self):
        super().__init__()
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = None

    @property
    def pipeline_layers(self):
        return self.layers[self.start_idx : self.end_idx]

    def pipeline(self, group):
        """Split layers across ranks proportionally based on available memory.

        On heterogeneous clusters (e.g. 256GB + 64GB + 48GB), equal splitting
        either wastes large nodes or OOMs small ones. This method queries each
        node's Metal working set size and assigns layers proportionally, with
        per-layer compute overhead factored in to avoid OOM.

        Falls back to equal splitting when memory info is unavailable.
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        n_layers = len(self.layers)

        layer_counts = self._compute_layer_split(n_layers, group)


        # Pipeline assigns in reverse: rank 0 = last layers, rank N = first
        start = sum(
            layer_counts[i]
            for i in range(self.pipeline_size)
            if i > self.pipeline_rank
        )
        count = layer_counts[self.pipeline_rank]
        self.start_idx = start
        self.end_idx = start + count
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx

    def pipeline_warmup(self):
        """Compile Metal shaders incrementally to avoid command buffer timeout.

        The first cold forward pass builds a compute graph spanning all local
        layers plus distributed ops whose shader compilation can exceed
        Metal's command buffer timeout.  This runs the layers in small
        batches first — each batch compiles only a few shaders, staying
        well within the timeout.
        """
        import sys
        from .base import create_attention_mask

        if self.pipeline_size <= 1:
            return

        pr = self.pipeline_rank
        ps = self.pipeline_size
        n = len(self.pipeline_layers)
        tokens = mx.array([[1, 2, 3]])

        print(f"[pipeline rank={pr}] warmup: {n} layers...",
              file=sys.stderr, flush=True)

        # Phase 1: Compile layer shaders locally (no distributed ops).
        # Each rank runs its own layers in chunks of 5 with independent
        # evals — no recv/send, so no cross-rank dependencies.
        h = self.embed_tokens(tokens)
        mask = create_attention_mask(h, None, return_array=True)
        mx.eval(h)
        for start in range(0, n, 5):
            h_local = self.embed_tokens(tokens)
            for i in range(start, min(start + 5, n)):
                h_local = self.pipeline_layers[i](h_local, mask, cache=None)
            mx.eval(h_local)

        # Phase 2: Sync all ranks, then run a full pipeline forward pass.
        # Shaders are compiled from Phase 1, so the graph executes fast
        # even though it spans all layers + recv/send/all_gather.
        mx.eval(mx.distributed.all_sum(mx.zeros(1)))
        h = self.embed_tokens(tokens)
        mask = create_attention_mask(h, None, return_array=True)
        if pr < ps - 1:
            h = mx.distributed.recv_like(h, pr + 1)
        for i in range(n):
            h = self.pipeline_layers[i](h, mask, cache=None)
        if pr != 0:
            h = mx.distributed.send(h, (pr - 1) % ps)
        if ps > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]
        h = self.norm(h)
        mx.eval(h, mx.distributed.all_sum(mx.zeros(1)))

        print(f"[pipeline rank={pr}] warmup: done",
              file=sys.stderr, flush=True)

    def _compute_layer_split(self, n_layers, group):
        """Determine how many layers each rank should get."""
        try:
            result = self._memory_proportional_split(n_layers, group)
            import sys
            print(f"[pipeline rank={self.pipeline_rank}] proportional split: {result}", file=sys.stderr, flush=True)
            return result
        except Exception as e:
            import sys
            print(f"[pipeline rank={self.pipeline_rank}] proportional split FAILED: {e}, using equal", file=sys.stderr, flush=True)
            return self._equal_split(n_layers)

    def _equal_split(self, n_layers):
        """Simple equal split (original behavior)."""
        layers_per_rank = n_layers // self.pipeline_size
        extra = n_layers - layers_per_rank * self.pipeline_size
        counts = []
        for i in range(self.pipeline_size):
            c = layers_per_rank + (1 if i < extra else 0)
            counts.append(c)
        return counts

    def _memory_proportional_split(self, n_layers, group):
        """Split layers proportionally based on available Metal working set."""
        import psutil

        if mx.metal.is_available():
            local_ws = mx.metal.device_info().get(
                "max_recommended_working_set_size",
                psutil.virtual_memory().total * 0.94,
            )
        else:
            local_ws = psutil.virtual_memory().total * 0.94

        local_model_bytes = sum(
            p.nbytes
            for _, p in tree_flatten(self.parameters())
            if isinstance(p, mx.array)
        )
        # Share model bytes across ranks — on lazy-loaded models, each rank
        # may report different sizes depending on which weight files exist locally.
        # Use the max across ranks as the authoritative total.
        all_bytes = mx.distributed.all_gather(
            mx.array([float(local_model_bytes)], dtype=mx.float32), group=group
        )
        mx.eval(all_bytes)
        # On lazy-loaded models, ranks may report different parameter sizes
        # depending on which weight files exist locally. Broadcast rank 0's
        # value as the reference since it always has full model access.
        total_model_bytes = float(all_bytes[0].item())
        bytes_per_layer = total_model_bytes / n_layers if n_layers > 0 else 1

        # Per-layer compute overhead covers KV cache, attention scores,
        # MoE activations, and MLX graph buffers
        compute_overhead = 1.5 * (1024**3)  # 1.5 GB
        embed_overhead = 2.5 * (1024**3)  # embeddings + lm_head

        max_local = max(
            1,
            int((local_ws - embed_overhead) / (bytes_per_layer + compute_overhead)),
        )
        local_capacity = max(float(max_local) * bytes_per_layer, bytes_per_layer)

        # Gather capacity from all ranks via collective
        all_cap = mx.distributed.all_gather(
            mx.array([local_capacity], dtype=mx.float32), group=group
        )
        # Materialize distributed gather (mx.eval is MLX graph materializer)
        mx.eval(all_cap)
        capacities = [float(all_cap[i].item()) for i in range(self.pipeline_size)]
        import sys
        print(
            f"[pipeline rank={self.pipeline_rank}] "
            f"local_cap={local_capacity/1e9:.1f}GB "
            f"all_cap={[round(c/1e9, 1) for c in capacities]} "
            f"bpl={bytes_per_layer/1e9:.2f}GB",
            file=sys.stderr, flush=True,
        )
        total_cap = sum(capacities)

        # Compute per-node max layers
        max_per_node = []
        for i in range(self.pipeline_size):
            ws_i = capacities[i] + (
                embed_overhead
                + max(1, int(capacities[i] / bytes_per_layer)) * compute_overhead
            )
            max_i = max(
                1,
                int((ws_i - embed_overhead) / (bytes_per_layer + compute_overhead)),
            )
            max_per_node.append(max_i)

        # Proportional split, capped at each node's max
        layer_counts = []
        assigned = 0
        for i in range(self.pipeline_size):
            if i == self.pipeline_size - 1:
                count = n_layers - assigned
            else:
                count = max(1, int(n_layers * capacities[i] / total_cap))
                count = min(count, max_per_node[i])
            layer_counts.append(count)
            assigned += count

        # Redistribute excess from small nodes to the largest
        largest_idx = capacities.index(max(capacities))
        if layer_counts[-1] > max_per_node[-1]:
            excess = layer_counts[-1] - max_per_node[-1]
            layer_counts[-1] = max_per_node[-1]
            layer_counts[largest_idx] += excess

        return layer_counts
