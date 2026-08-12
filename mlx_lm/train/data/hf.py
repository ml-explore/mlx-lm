# Copyright © 2026 Apple Inc.

from mlx_lm.train.data.batching import tokenized_data


def hf_data(dataset, text_key="text", start_sample_idx=0):
    """Documents from a streaming dataset, numbered so a position can be named."""
    for sample_idx, record in enumerate(dataset):
        if sample_idx < start_sample_idx:
            continue
        yield {
            "text": record[text_key],
            "file_name": None,
            "sample_idx": sample_idx,
        }


def load_hf(
    tokenizer,
    rank,
    size,
    dataset,
    name=None,
    split="train",
    data_files=None,
    text_key="text",
    shuffle_buffer=None,
    seed=0,
    start_sample_idx=0,
):
    """This rank's documents, streamed from the hub.

    Args:
        tokenizer: Applied to ``text_key``.
        rank (int): This rank's index; it reads its own slice of the shards.
        size (int): The number of ranks reading the dataset.
        dataset (str): A hub id such as ``"allenai/dolma3_mix-6T"``, or a builder
            name such as ``"json"`` alongside ``data_files``.
        shuffle_buffer (int, optional): Draw from a buffer this size rather than
            in order. Applied after this rank's shards are chosen, so the buffer
            holds only its own documents. Default: ``None``.
    """
    from datasets import load_dataset

    ds = load_dataset(
        dataset,
        name=name,
        split=split,
        data_files=data_files,
        streaming=True,
    )

    if size > 1:
        if ds.num_shards >= size:
            ds = ds.shard(num_shards=size, index=rank, contiguous=False)
        else:
            from datasets.distributed import split_dataset_by_node

            ds = split_dataset_by_node(ds, rank=rank, world_size=size)

    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return tokenized_data(tokenizer, hf_data(ds, text_key, start_sample_idx))
