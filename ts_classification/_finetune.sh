#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
# DEFAULT_CLASSES=("apple" "banana" "asparagus" "avocado")
DEFAULT_CLASSES="all"
DEFAULT_FEATURES="all"   # "all" = auto-detect all numeric columns, or e.g. ("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")

WANDB_PROJECT="smell-net"
WANDB_RUN_NAME="all_test"   # leave empty to let WandB auto-generate a name

python "${SCRIPT_DIR}/_finetune_run.py" \
  --mode train \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --model "timesnet" \
  --seq-len 512 \
  --batch-size 2 \
  --epochs 7 \
  --learning-rate 1e-3 \
  --val-frequency 1 \
  --eval-split "test" \
  --eval-at-end \
  --save-dir "${PROJECT_ROOT}/smell_model_ckpts" \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  "$@"
