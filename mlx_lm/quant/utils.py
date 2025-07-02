# Copyright © 2025 Apple Inc.

from pathlib import Path

import mlx.core as mx


def load_data(tokenizer, num_samples: int, sequence_length: int) -> mx.array:
    save_dir = Path.home() / ".cache/mlx-lm/calibration_v5.txt"
    if not save_dir.exists():
        from urllib import request

        save_dir.parent.mkdir(parents=True, exist_ok=True)
        url = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt"
        request.urlretrieve(url, save_dir)
    with open(save_dir) as fid:
        texts = fid.read()
    tokens = tokenizer.encode(texts, return_tensors="mlx")[0]

    # select random non-overlapping chunks
    tokens = tokens[: (tokens.size // sequence_length) * sequence_length]
    tokens = tokens.reshape(-1, sequence_length)
    segments = mx.random.permutation(tokens.shape[0])
    if num_samples > 0:
        segments = segments[:num_samples]
    return tokens[segments]
