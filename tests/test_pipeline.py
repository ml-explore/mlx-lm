# Copyright © 2026 Apple Inc.

from mlx_lm.models.pipeline import PipelineMixin


class _Group:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _Pipeline(PipelineMixin):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


def test_pipeline_partition_covers_non_divisible_layer_count_exactly_once():
    assigned = []
    for rank in range(3):
        pipeline = _Pipeline(list(range(10)))
        pipeline.pipeline(_Group(rank, 3))
        assigned.extend(
            layer for layer in pipeline.pipeline_layers if layer is not None
        )

    assert sorted(assigned) == list(range(10))
    assert len(assigned) == len(set(assigned))
