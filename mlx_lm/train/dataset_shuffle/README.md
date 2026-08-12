# Dataset mirror and shuffle

Two tools for preparing a large sharded corpus that is sorted by topic (for
example `allenai/dolma3_mix-6T`):

1. `mirror_dataset.py` copies it into your own S3 bucket.
2. `shuffle_dataset.py` shuffles it in two stages to make sampling uniform.

`shuffle_dataset` can read `hf://` directly, so the mirror is a convenience, not a
prerequisite -- you can go straight from the Hub. 

Both are sharded the same way -- one process per `--worker-id`, all in parallel,
no coordination between workers.

## Install

```sh
pip install -r requirements.txt
```

## 1. Mirror (optional)

```sh
python mirror_dataset.py \
    --input  hf://datasets/allenai/Dolci-Think-SFT-7B/data/ \
    --output s3://my-bucket/dolci-think-sft/ \
    --num-workers 32 --worker-id 0
```

**Transfers resume.** A file already at the destination is skipped, so a worker
that dies part way through is just rerun, and topping up a mirror later costs one
existence check per file.

| flag | effect |
| --- | --- |
| `--overwrite` | recopy files that already exist |
| `--dry-run` | print the plan, transfer nothing |
| `--check` | validate endpoint, bucket and input, then exit |

Unlike the shuffle, the mirror copies every file it finds, not just those with a
data extension, so dataset cards and metadata will be copied too.

## 2. Shuffle

```
stage 1   Worker w reads its 1/W slice of the input files, shuffles the records,
          and scatters them into W buckets. Bucket (w, j) holds a random ~1/W² of
          all records.

              worker 0 ──> bucket (0,0) (0,1) (0,2) ...
              worker 1 ──> bucket (1,0) (1,1) (1,2) ...
              worker 2 ──> bucket (2,0) (2,1) (2,2) ...
                                    │     │     │
stage 2   Worker j reads column j ──┘     │     │  -- one bucket from every
          stage-1 worker, together a uniform 1/W sample of the whole corpus --
          shuffles again, and writes its final shards.
```

Run every worker of stage 1, wait for all of them, then run every worker of
stage 2.

```sh
# stage 1
python shuffle_dataset.py --stage 1 --worker-id 0 --num-workers 32 \
    --input  s3://my-bucket/dolma-mix/data/ \
    --output s3://my-bucket/dolma-mix-shuffled/data/

# stage 2: same W, plus how many final files each worker writes
python shuffle_dataset.py --stage 2 --worker-id 0 --num-workers 32 \
    --num-output-shards 32 \
    --output s3://my-bucket/dolma-mix-shuffled/data/
```

Each record gets a `source_file` field, `<last three path components>:<line
number>`, recording where it came from. `--no-source-column` turns this off,
which is noticeably faster for JSON Lines.

### Choosing W

A worker holds its whole slice in memory, uncompressed, plus the shard it is
writing. Pick `--num-workers` so that

```
uncompressed corpus bytes / W   <   RAM per machine, with room to spare
```

A 50B-token Dolma mix was shuffled with `W=32` on 256 GB machines. Peak memory is
`W` times lower than a single-machine shuffle but no lower, so `W` is a memory
knob first and a parallelism knob second. Running several workers on one machine
multiplies its memory use.

`--num-workers` must be identical in both stages and must not exceed the number
of input files. Total final files is `W * --num-output-shards`.

### Output layout

```
<output>00000-of-00032/00000-of-00032.jsonl.gz    <- W x --num-output-shards
<output>00000-of-00032/00001-of-00032.jsonl.gz       final files
...
<output>_tmp/...                                  <- stage 1's buckets
```

Delete `<output>/_tmp/` once you are happy with the output; nothing reads it
after stage 2.

## End to end

Shuffle straight from the Hub -- one script, no mirror:

```sh
INPUT=hf://datasets/allenai/Dolci-Think-SFT-7B/ \
OUTPUT=s3://smollm/dolci-shuffled/data/ \
NUM_WORKERS=32 ./dataset_shuffle/scripts/shuffle_dataset.sh
```

Or mirror first. Separate scripts:

```sh
# 1. mirror -- network-bound, so oversubscribe it
INPUT=hf://datasets/allenai/Dolci-Think-SFT-7B/ \
OUTPUT=s3://smollm/dolci-shuffled/data/ \
  ./dataset_shuffle/scripts/shuffle_dataset.sh

# 2. shuffle -- memory-bound, so NUM_WORKERS is a memory budget
INPUT=s3://my-bucket/dolmino-flan/ \
OUTPUT=s3://my-bucket/dolmino-flan-shuffled/data/ \
NUM_WORKERS=32 ./scripts/shuffle_dataset.sh
```

## Paths

`--input` is a prefix ending in `/` (listed recursively), an `hf://` Hub dataset,
or a text file of paths relative to its own directory. A file list may name
`hf://` or `s3://` URIs directly, one per line, which is how you shuffle a
hand-picked subset or mix files from several places.

Data files are picked out of a listing by extension, and `--format` is inferred
from them: `.jsonl`, `.json`, either with an optional `.gz`/`.zst`/`.zstd`, or
`.parquet`. Everything else is ignored, so a `README` alongside the data is
harmless -- but a metadata file ending in `.json` would be read as data, so for a
prefix like that pass an explicit file list instead.


## Credentials and non-AWS stores

Credentials come from the environment, the usual boto3 chain:
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, a profile, or an instance role.

On real AWS, boto3 derives the address from your region and there is nothing to configure:

```
region us-west-2  ->  https://s3.us-west-2.amazonaws.com
```

For any other store compatibiile with S3 the endpoint should be set:

```sh
export S3_ENDPOINT_URL=https://my-object-store.example.com
```
