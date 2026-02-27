#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
# DEFAULT_CLASSES=("apple" "banana" "asparagus" "avocado")
DEFAULT_CLASSES="all"
DEFAULT_FEATURES=("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")

WANDB_PROJECT="smell-net"
WANDB_RUN_NAME="all_test"   # leave empty to let WandB auto-generate a name

# --- Validation ---
VAL_FREQUENCY=1       # validate every N epochs (1 = every epoch)
EVAL_AT_END=true      # run a final evaluation after the last training epoch
EVAL_SPLIT="test"     # which split to use for the end-of-training eval: val | test

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
  --val-frequency "${VAL_FREQUENCY}" \
  --eval-split "${EVAL_SPLIT}" \
  ${EVAL_AT_END:+--eval-at-end} \
  --save-dir "${PROJECT_ROOT}/smell_model_ckpts" \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  "$@"
