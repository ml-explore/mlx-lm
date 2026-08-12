#!/bin/bash
# Mirror hf dataset into your own bucket.
#
# Copies every file from INPUT hf location to s3 OUTPUT.
# Example for allenai/dolma3_mix-6T:
#   
#   INPUT=hf://datasets/allenai/dolma3_mix-6T \
#   OUTPUT=s3://my-bucket/dolma3_mix-6T/ ./mirror_dataset.sh


set -euo pipefail

INPUT=${INPUT:-hf://datasets/allenai/dolma3_mix-6T}
OUTPUT=${OUTPUT:-s3://my-bucket/dolma3_mix-6T/}

# Each worker handles one file at a time, so NUM_WORKERS only has to be large enough to
# split the work; PARALLEL is how many actually run at once.
NUM_WORKERS=${NUM_WORKERS:-32}
PARALLEL=${PARALLEL:-16}

# For an S3-compatible store that is not AWS. Leave
# empty for real AWS, where the endpoint comes from your region. 
export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-}

HERE=$(cd -- "$(dirname -- "$0")" && pwd)

echo "=== mirror: ${INPUT} -> ${OUTPUT} (${NUM_WORKERS} workers, ${PARALLEL} at a time) ==="

# Validate once up front: a bad endpoint, a missing bucket or an empty input
# fails here in about a second, rather than being rediscovered and reported
# separately by every one of NUM_WORKERS processes.
python "${HERE}/../mirror_dataset.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --check

seq 0 $((NUM_WORKERS - 1)) | xargs -P "${PARALLEL}" -I{} \
  python "${HERE}/../mirror_dataset.py" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --num-workers "${NUM_WORKERS}" \
    --worker-id {}

cat <<EOF

Mirrored ${INPUT}
      to ${OUTPUT}

Now shuffle it:
  INPUT=${OUTPUT} OUTPUT=<destination>/data/ ${HERE}/shuffle_dataset.sh

Keep the mirror if you plan to reshuffle -- rerunning the shuffle rereads it.
EOF
