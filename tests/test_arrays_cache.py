# Copyright © 2026 Mark Smeltzer

import pathlib
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache


class TestArraysCache(unittest.TestCase):
    def test_advance_bounds_metadata_graph(self):
        cache = ArraysCache(2, left_padding=[2])
        cache.prepare(lengths=[3])
        cache[0] = mx.array([0])
        cache[1] = mx.array([0])

        for _ in range(256):
            cache[0] = cache[0] + 1
            cache.advance(1)
            mx.eval(cache[0])

        with tempfile.TemporaryDirectory() as directory:
            for name, metadata in (
                ("lengths", cache.lengths),
                ("left-padding", cache.left_padding),
            ):
                graph_path = pathlib.Path(directory, f"arrays-cache-{name}.dot")
                mx.export_to_dot(str(graph_path), metadata)
                edge_count = graph_path.read_text(encoding="utf-8").count("->")
                self.assertLessEqual(edge_count, 8, f"{name} graph has {edge_count} edges")

        self.assertEqual(cache[0].item(), 256)
        self.assertEqual(cache.lengths.item(), 3 - 256)
        self.assertEqual(cache.left_padding.item(), 2 - 256)


if __name__ == "__main__":
    unittest.main()
