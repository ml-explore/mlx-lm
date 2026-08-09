# Copyright © 2026 Apple Inc.

from mlx_lm import perplexity


class _Dataset:
    values = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def process(self, value):
        return value, None


def test_load_data_all_samples_consumes_finite_dataset_once(monkeypatch):
    monkeypatch.setattr(
        perplexity, "load_dataset", lambda args, tokenizer: (_Dataset(), None, None)
    )
    monkeypatch.setattr(
        perplexity.np.random,
        "permutation",
        lambda size: perplexity.np.arange(size),
    )

    data = perplexity.load_data(None, "fixture", num_samples=-1, sequence_length=4)

    assert data.shape == (2, 4)
    assert data.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
