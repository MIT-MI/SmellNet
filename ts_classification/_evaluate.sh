#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
DEFAULT_CLASSES=("banana" "apple")
DEFAULT_FEATURES=("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")
CHECKPOINT_PATH="${PROJECT_ROOT}/artifacts/classifier_best.pt"
METADATA_PATH="${PROJECT_ROOT}/artifacts/metadata.json"

python "${PROJECT_ROOT}/_run.py" \
  --mode eval \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --seq-len 512 \
  --checkpoint "${CHECKPOINT_PATH}" \
  --metadata "${METADATA_PATH}" \
  "$@"
