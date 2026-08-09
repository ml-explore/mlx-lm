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

    def pipeline(self, group):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        num_layers = len(self.layers)
        layers_per_rank, extra = divmod(num_layers, self.pipeline_size)
        # Convert the reverse pipeline rank to the corresponding forward
        # partition, then use a prefix sum. Multiplying by a rank-local shard
        # size creates gaps whenever the division has a remainder.
        partition = self.pipeline_size - self.pipeline_rank - 1
        self.start_idx = partition * layers_per_rank + min(partition, extra)
        layers_per_rank += partition < extra
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        # Keep the layer numbers the same for model loading
        self.layers[: self.start_idx] = [None] * self.start_idx
