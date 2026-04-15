# Copyright © 2025 Apple Inc.

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mlx_lm.examples.rag import cosine_similarity, retrieve


class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        # Identical vectors should have similarity of 1.0
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0, places=5)

    def test_orthogonal_vectors(self):
        # Orthogonal vectors should have similarity of 0.0
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_opposite_vectors(self):
        # Opposite vectors should have similarity of -1.0
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)


class TestRetrieve(unittest.TestCase):

    def setUp(self):
        self.documents = [
            "MLX is an array framework for machine learning on Apple silicon, developed by Apple.",
            "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
            "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce energy.",
        ]

    def test_retrieves_most_similar_document(self):
        # Mock get_embedding to return controlled vectors
        # Question embedding is closest to document 0
        embeddings = {
            "what is mlx?": np.array([1.0, 0.0, 0.0]),
            self.documents[0]: np.array([0.9, 0.1, 0.0]),  # most similar
            self.documents[1]: np.array([0.0, 1.0, 0.0]),
            self.documents[2]: np.array([0.0, 0.0, 1.0]),
        }

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "mlx_lm.examples.rag.get_embedding",
            side_effect=lambda text, m, t: embeddings[text],
        ):
            result = retrieve(
                "what is mlx?", self.documents, mock_model, mock_tokenizer
            )

        self.assertEqual(result, self.documents[0])

    def test_single_document_always_returned(self):
        # With a single document, it should always be returned
        single_doc = ["Only document."]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "mlx_lm.examples.rag.get_embedding",
            return_value=np.array([1.0, 0.0]),
        ):
            result = retrieve("any question", single_doc, mock_model, mock_tokenizer)

        self.assertEqual(result, single_doc[0])


if __name__ == "__main__":
    unittest.main()
