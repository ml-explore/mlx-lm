# Copyright © 2025 Apple Inc.

import mlx.core as mx


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
            base = len(self.layers) // self.pipeline_size
            extra = len(self.layers) - base * self.pipeline_size
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
