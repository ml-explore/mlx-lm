# Copyright © 2025 Apple Inc.


def _rank_sizes(dim, N, block=1):
    # Split dim into N ranks as evenly as possible: the first `extra`
    # ranks get one extra unit (or block, for group_size-aware quantized
    # splits), the rest get the base amount. Reduces to an exactly even
    # split whenever dim % (N * block) == 0.
    n_blocks = dim // block
    base = n_blocks // N
    extra = n_blocks - base * N
    return [(base + (1 if i < extra else 0)) * block for i in range(N)]


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

    def pipeline(self, group, split=None):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        if split is None:
            # Even split; the low ranks get the extra layers.
            base, extra = divmod(len(self.layers), self.pipeline_size)
            split = [base + (1 if r < extra else 0) for r in range(self.pipeline_size)]
        if len(split) != self.pipeline_size:
            raise ValueError(
                f"split has {len(split)} entries for group size {self.pipeline_size}"
            )
        if any(s <= 0 for s in split) or sum(split) != len(self.layers):
            raise ValueError(
                f"split {split} must be positive and sum to {len(self.layers)} layers"
            )
        # Rank r runs the layers after the layers of ranks r+1 ... size-1.
        self.start_idx = sum(split[self.pipeline_rank + 1 :])
        self.end_idx = self.start_idx + split[self.pipeline_rank]
        self.layers = self.layers[: self.end_idx]
        # Keep the layer numbers the same for model loading
        self.layers[: self.start_idx] = [None] * self.start_idx
