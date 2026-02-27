#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
DEFAULT_CLASSES=("apple" "banana" "asparagus" "avocado")
DEFAULT_FEATURES=("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")

WANDB_PROJECT="smell-net"
WANDB_RUN_NAME=""   # leave empty to let WandB auto-generate a name

python "${SCRIPT_DIR}/_finetune_run.py" \
  --mode train \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --model "tscmamba" \
  --seq-len 512 \
  --batch-size 16 \
  --epochs 30 \
  --learning-rate 1e-3 \
  --save-dir "${PROJECT_ROOT}/artifacts" \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  "$@"
