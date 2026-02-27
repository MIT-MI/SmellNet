#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
DEFAULT_CLASSES=("banana" "apple")
DEFAULT_FEATURES=("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")
CHECKPOINT_PATH="${PROJECT_ROOT}/artifacts/classifier_best.pt"
METADATA_PATH="${PROJECT_ROOT}/artifacts/metadata.json"

WANDB_PROJECT="smell-net"
WANDB_RUN_NAME=""   # leave empty to let WandB auto-generate a name

# --- Evaluation split ---
EVAL_SPLIT="test"   # which split to evaluate on: val | test

python "${SCRIPT_DIR}/_finetune_run.py" \
  --mode eval \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --seq-len 512 \
  --checkpoint "${CHECKPOINT_PATH}" \
  --metadata "${METADATA_PATH}" \
  --eval-split "${EVAL_SPLIT}" \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  "$@"
