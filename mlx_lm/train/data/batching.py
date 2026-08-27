# Copyright © 2026 Apple Inc.

import multiprocessing as mp

import ml_collections
import numpy as np

import mlx_lm.train.data as data

STAGES = ("pre", "mid")
SOURCES = ("hf", "s3")


def dolma(stage="pre", source="hf"):
    """The dolma corpus for a stage, read from the hub or from S3."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    corpora = data.hf.DOLMA if source == "hf" else data.s3.DOLMA
    return ml_collections.ConfigDict({"source": source, **corpora[stage]})


def get_documents(dataset, tokenizer, mesh, data_state, seed=0):
    resume_sample_idx = data_state.get("sample_idx", 0)

    if dataset.source == "hf":
        documents = data.load_hf(
            tokenizer,
            mesh.world.rank,
            mesh.world.size,
            dataset=dataset.name,
            name=dataset.get("subset") or None,
            split=dataset.get("split", "train"),
            data_files=dataset.get("data_files"),
            text_key=dataset.get("text_key", "text"),
            shuffle_buffer=dataset.get("shuffle_buffer"),
            seed=seed,
            start_sample_idx=resume_sample_idx,
        )
    elif dataset.source == "s3":
        documents = data.load_s3(
            tokenizer,
            mesh.world.rank,
            mesh.world.size,
            uri=dataset.uri,
            num_groups=dataset.num_groups,
            shards_per_group=dataset.get("shards_per_group", 32),
            suffix=dataset.get("suffix", ".json.gz"),
            start_file_name=data_state.get("file_name"),
            start_sample_idx=resume_sample_idx,
        )
    else:
        raise ValueError(
            f"unknown config.dataset.source {dataset.source!r}; expected 's3', "
            "with config.dataset.uri an s3:// prefix, or 'hf', with "
            "config.dataset.name a Hugging Face id"
        )
    return documents


def prefetch(iterator):
    sentinel = None

    def f(iterator, queue, sentinel):
        for sample in iterator:
            queue.put(sample)
        queue.put(sentinel)

    queue = mp.Queue(100)
    process = mp.get_context("fork").Process(target=f, args=(iterator, queue, sentinel))
    process.start()
    try:
        while True:
            sample = queue.get()
            if sample == sentinel:
                break
            yield sample
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def tokenized_data(tokenizer, dataset):
    for d in dataset:
        tokens = tokenizer.encode(d["text"], add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        yield {
            "input_ids": tokens,
            "text": d["text"],
            "file_name": d["file_name"],
            "sample_idx": d.get("sample_idx", 0),
        }


def iterate_batches(
    dataset,
    context_size,
    batch_size,
    max_batches=None,
    resume_state=None,
):
    """
    Simply concatenate documents until the batch is full.
    """
    seq_len = context_size + 1
    max_batches = max_batches or float("inf")
    resume_state = resume_state or {}
    d_next = list(resume_state.get("d_next", []))
    last_file_name = resume_state.get("file_name")
    next_sample_idx = resume_state.get("sample_idx", 0)
    resume_batch_idx = resume_state.get("batch_idx", 0)
    batch_num = 0
    while batch_num < max_batches:
        batch = np.empty((batch_size * seq_len), np.int32)
        i = 0
        while i < len(batch):
            if len(d_next) > 0:
                d = d_next
                d_next = []
            else:
                sample = next(dataset, None)
                if sample is None:
                    break
                last_file_name = sample.get("file_name")
                next_sample_idx = sample.get("sample_idx", 0) + 1
                d = sample["input_ids"]
            e = i + len(d)
            if e > len(batch):
                trim = e - len(batch)
                d_next = d[-trim:]
                d = d[:-trim]
                e = len(batch)
            batch[i:e] = d
            i += len(d)
        if i < len(batch):
            break
        batch_num += 1
        yield {
            "input_ids": batch.reshape(batch_size, seq_len),
            "mask": None,
            "_data_state": {
                "file_name": last_file_name,
                "sample_idx": next_sample_idx,
                "batch_idx": resume_batch_idx + batch_num,
                "d_next": list(d_next),
            },
        }
