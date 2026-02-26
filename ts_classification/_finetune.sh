#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
DEFAULT_CLASSES=("diego_mixtures" "apple")
DEFAULT_FEATURES=("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")

python "${PROJECT_ROOT}/ts_classification/_run.py" \
  --mode train \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --seq-len 512 \
  --batch-size 16 \
  --epochs 30 \
  --learning-rate 1e-3 \
  --val-split 0.2 \
  --save-dir "${PROJECT_ROOT}/artifacts" \
  "$@"
