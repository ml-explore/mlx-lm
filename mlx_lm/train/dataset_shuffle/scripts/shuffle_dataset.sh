#!/bin/bash
# Shuffle a dataset with the two-stage shuffle.
#
# Works for both the JSON Lines pretraining mixes and the Parquet SFT sets --
# --format is inferred from the input file extensions. INPUT can be a local
# directory, an s3:// prefix, or an hf:// dataset on the Hugging Face Hub, though
# for anything you will shuffle more than once, mirror it into your own bucket
# first with ./mirror_dataset.sh and point INPUT at that.
#
#   INPUT=s3://my-bucket/dolci-think-sft/ \
#   OUTPUT=s3://my-bucket/dolci-think-sft-shuffled/data/ ./shuffle_dataset.sh
set -euo pipefail

INPUT=${INPUT:-s3://my-bucket/dolci-think-sft/}
OUTPUT=${OUTPUT:-s3://my-bucket/dolci-think-sft-shuffled/data/}

# W: each worker holds ~1/W of the corpus in memory. Pick W so that
# corpus_bytes / W fits in one machine's RAM

NUM_WORKERS=${NUM_WORKERS:-8}

# Final files per stage-2 worker, so NUM_WORKERS * this many files in total.
NUM_OUTPUT_SHARDS=${NUM_OUTPUT_SHARDS:-16}

# Workers to run at once on this machine. Peak memory is roughly
# PARALLEL * corpus_bytes / NUM_WORKERS, so raise it only if that fits.
PARALLEL=${PARALLEL:-1}

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-}

HERE=$(cd -- "$(dirname -- "$0")" && pwd)

run_stage () {
  local stage=$1
  echo "=== stage ${stage}: ${NUM_WORKERS} workers, ${PARALLEL} at a time ==="
  seq 0 $((NUM_WORKERS - 1)) | xargs -P "${PARALLEL}" -I{} \
    python "${HERE}/../shuffle_dataset.py" \
      --stage "${stage}" \
      --input "${INPUT}" \
      --output "${OUTPUT}" \
      --num-workers "${NUM_WORKERS}" \
      --num-output-shards "${NUM_OUTPUT_SHARDS}" \
      --worker-id {}
}

# Every stage-1 worker must finish before any stage-2 worker starts: stage 2
# reads one bucket from each of them
run_stage 1
run_stage 2

cat <<EOF

Wrote ${NUM_WORKERS} x ${NUM_OUTPUT_SHARDS} shards under ${OUTPUT}
Once you have checked the output, delete the intermediate buckets under
${OUTPUT}_tmp/ -- for S3 that is:
  aws s3 rm --recursive ${OUTPUT}_tmp/
EOF
